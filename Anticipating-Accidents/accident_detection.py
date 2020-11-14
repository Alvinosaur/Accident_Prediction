import torch
import torch.nn as nn


"""
Possible architecture changes:
- process k frames at a time, not just 1 frame
- bidirectional
- apply MLP's to features, not just affine transformation
- change to not have preset number of objects, handle any N objects
- play with LSTM params, or variants (ex: GRU)

Implementation Details:
- objects can be diff size, so how to transform them into features? pass
    entire image, but mask object so only its pixels are non-zero. this
    way, same weights of same size can be applied to each object
    - but in this case, weights depend on spatial location of an object,
        so this is how transforming object into a feature encodes its
        position, not just the object pixels, which are uninformative
- paper uses hyperparam for number of objects in image. What if there
    are less than num_objs in a given image? those extra remaining images
    are completely blacked out, and after applying linear transform, we
    just black out the entire feature so has no effect
    - but if some objects are not visible and their pixels are all
        blacked out, then why do we need to use the post-transformation
        mask to zero everything out?
- paper applies mask after softmax... doesn't this break the law of total
  probability? Shouldn't we mask out nonexistent objects before the softmax?

Possible Bugs:
- try with no frame weights
- try alphas = torch.softmax(torch.multiply(alphas, mask), dim=1) line 104
- prev_output, (hidden_state, cell_state) = self._lstm(
                fusion, (hidden_state, cell_state))
"""


class AccidentLoss(object):
    def __init__(self, n_frames, device):
        self.n_frames = n_frames
        pos_weights = torch.exp(
            - torch.arange(self.n_frames - 1, -1, -1) / 20.0).view(-1, 1)
        neg_weights = torch.ones((n_frames, 1))
        # (n_frames x 2)
        self.frame_weights = torch.cat([neg_weights, pos_weights], dim=1)
        self.frame_weights = self.frame_weights.to(device)
        self.frame_weights.requires_grad = False
        self.log_softmax = torch.nn.LogSoftmax(dim=2)
        self.nll_loss = torch.nn.NLLLoss()

    def __call__(self, logits, labels):
        # (n_frames x B x 2)
        loss = self.log_softmax(logits)
        # (n_frames x B x 2) multiply each frame's outputs with specific weight
        loss = torch.mul(self.frame_weights, loss)
        # (n_frames*B x 2) following NLLLoss's expected input of (minibatch, C)
        loss = loss.view(-1, 2)
        labels = labels.view(-1)
        # compute average loss over all frames of entire batch
        loss = self.nll_loss(loss, labels)
        return loss


class Fattn(nn.Module):
    def __init__(self, lstm_hidden_dim, obj_feat_dim):
        """Calculates alpha = softmax(f_attn(h_t-1, a_t)), which
        assigns

        Args:
            lstm_hidden_dim ([type]): [description]
            obj_feat_dim ([type]): [description]

        Returns:
            [type]: [description]
        """
        super().__init__()
        # linear transform of previous hidden state
        self._lstm_hidden_dim = lstm_hidden_dim
        self._obj_feat_dim = obj_feat_dim
        self.hidden_linear = nn.Linear(
            self._lstm_hidden_dim, self._obj_feat_dim, bias=False)
        self.combined_linear = nn.Linear(
            self._obj_feat_dim, 1, bias=False)

    def forward(self, a, hprev, mask):
        """[summary]

        Args:
            a (Tensor): (B x K-1 x obj_feat_dim) feature vecs of diff objs
            hprev (Tensor): (B x lstm_hidden_dim)
            mask (Tensor): (B x K-1)
        """
        # possibly perform some transform here to combine all hidden layer units
        hprev = torch.squeeze(hprev)
        # (B x 1 x obj_feat_dim)
        hprime = torch.unsqueeze(self.hidden_linear(hprev), 1)
        e = torch.tanh(hprime + a)
        # (B x K-1 x obj_feat_dim) -> (B x K-1 x 1)
        alphas = self.combined_linear(e)
        # calculate probability/importance of each K-1 object
        # mask out any features that are non-existent
        alphas = torch.softmax(torch.mul(alphas, mask), dim=1)
        # alphas = torch.mul(torch.softmax(alphas, dim=1), mask)
        # assert(torch.sum(alphas, axis=1) == ones)
        # probability of each obj feature should sum to 1 for a batch
        # but after post-multiplying by mask, this isn't true anymore
        return alphas


class AccidentDetection(nn.Module):
    def __init__(self, img_dim, n_hidden_layers, img_feat_dim, obj_feat_dim, lstm_hidden_dim, device, lstm_dropout=0):
        """Main module encapsulating accident detection pipeline. Given a
        video sequence of images, processes one frame at a time. Output for a
        given frame is a (1 x 2) [1-p(accident), p(accident)].

        Args:
            img_dim (int): size of flattened image
            n_hidden_layers (int): number of hidden layers in LSTM
            img_feat_dim (int): size of processed image feature
            obj_feat_dim (int): size of processed object feature
        """
        super().__init__()
        self._img_dim = img_dim
        self._obj_dim = img_dim  # entire image masked out except for object
        self._n_hidden_layers = n_hidden_layers
        self._img_feat_dim = img_feat_dim
        self._obj_feat_dim = obj_feat_dim
        self._lstm_hidden_dim = lstm_hidden_dim
        self._num_dir = 1  # num directions, 2 if bidirectional
        self._num_hidden_states = self._num_dir * self._n_hidden_layers
        self.device = device

        self._img_to_feat = nn.Linear(self._img_dim, self._img_feat_dim)
        self._obj_to_feat = nn.Linear(self._obj_dim, self._obj_feat_dim)
        self._obj_to_feat2 = nn.Linear(self._obj_feat_dim, self._img_feat_dim)
        # output = [1-prob(accident), prob(accident)]
        self._out_to_pred = nn.Linear(self._lstm_hidden_dim, 2)
        self._lstm = nn.LSTM(
            input_size=self._img_feat_dim + self._obj_feat_dim,
            hidden_size=self._lstm_hidden_dim,
            num_layers=self._n_hidden_layers,
            batch_first=False,
            dropout=lstm_dropout)
        self._fattn = Fattn(self._lstm_hidden_dim, self._img_feat_dim)

    def forward(self, x):
        """Forward pass

        Args:
            x (Tensor): B x N x K x D
                B = Batch size
                N = num image frames per entry
                K = 1 + num objects to focus on
                D = input feature dimension (self._img_dim)
                D_i = image feature dim
                D_o = obj feature dim
                D_o2 = 2nd obj feature dim
        """
        B, N, K, D = x.shape
        # all zeros for an obj index in a frame of a specific batch if that obj
        # isn't present
        # no mask for first of K since that represents entire image, not an obj
        # (B x N x K-1 x 1)
        zeros = torch.zeros((B, N, K - 1, 1)).to(self.device)
        obj_mask = torch.sum(x[:, :, 1:], dim=-1, keepdim=True)
        obj_mask = torch.isclose(obj_mask, zeros, atol=1e-06)
        obj_mask = (obj_mask != True).float()

        # transform full image input vec into img feature
        # (B x N x D_i)
        img_feat = self._img_to_feat(x[:, :, 0, :])

        # transform each obj input vec into obj feature
        # (B x N x K-1 x D_o)
        obj_feat = self._obj_to_feat(x[:, :, 1:, :])

        # mask out any obj features where obj isn't present
        # (B x N x K-1, D_o) = (B x N x K-1 x D_o) * (B x N x K-1 x 1) < brdcst
        obj_feat = torch.mul(obj_feat, obj_mask)
        # 2nd affine transform
        # (B x N x K-1, D_o) -> (B x N x K-1, D_i)
        obj_feat = self._obj_to_feat2(obj_feat)

        # intialize LSTM hidden state and
        hidden_state = torch.zeros(
            (self._num_hidden_states, B, self._lstm_hidden_dim)).to(self.device)
        cell_state = torch.zeros(
            (self._num_hidden_states, B, self._lstm_hidden_dim)).to(self.device)
        prev_output = torch.zeros(
            (1, B, self._lstm_hidden_dim)).to(self.device)

        # track all info
        all_alphas = []
        all_predictions = []

        for fi in range(N):
            # (B x K-1 x D_i)
            cur_obj_feat = obj_feat[:, fi, :, :]
            # (B x D_i)
            cur_img_feat = img_feat[:, fi, :]
            # (B x K-1 x 1)
            cur_obj_mask = obj_mask[:, fi, :]

            # (B x K-1)
            alphas = self._fattn(cur_obj_feat, prev_output, cur_obj_mask)
            # weighted each object feature by its attention alphas
            # (B x K-1 x D_i)
            w_obj_feat = torch.mul(cur_obj_feat, alphas)
            # sum up all K-1 features to produce weighted sum
            # (B x D_i)
            w_obj_feat = torch.sum(w_obj_feat, dim=1)

            # (B x 2*D_i)
            fusion = torch.cat([cur_img_feat, w_obj_feat], dim=1)
            # (1 x B x 2*D_i) since lstm takes (seq_len, batch, input_size)
            fusion = torch.unsqueeze(fusion, dim=0)
            prev_output, (hidden_state, cell_state) = self._lstm(
                fusion, (hidden_state, cell_state))

            # possible combine sequence of outputs
            prev_output = torch.squeeze(prev_output, dim=0)

            # (B x H)
            logits = self._out_to_pred(prev_output)
            # predictions = torch.softmax(logits, dim=1)
            predictions = logits  # CrossEntropyLoss already applies LogSoftmax

            # save all outputs
            all_alphas.append(alphas)
            all_predictions.append(predictions)

        # (N x B x 2) --> (B x N x 2)
        all_predictions = torch.stack(all_predictions).transpose(1, 0)
        return all_alphas, all_predictions
