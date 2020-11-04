from statistics import mode, mean

import torch
import torch.nn.functional as F


class SaveValues():
    def __init__(self, m):
        # register a hook to save values of activations and gradients
        self.activations = None
        self.gradients = None
        self.forward_hook = m.register_forward_hook(self.hook_fn_act)
        self.backward_hook = m.register_backward_hook(self.hook_fn_grad)

    def hook_fn_act(self, module, input, output):
        self.activations = output

    def hook_fn_grad(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def remove(self):
        self.forward_hook.remove()
        self.backward_hook.remove()


class RAM3d(object):
    """ Regression Activation Mapping """

    def __init__(self, model, target_layer):
        """
        Args:
            model: a base model to get RAM3d which have global pooling and fully connected layer.
            target_layer: conv_layer before Global Average Pooling
        """

        self.model = model
        self.target_layer = target_layer

        # save values of activations and gradients in target_layer
        self.values = SaveValues(self.target_layer)


class GradRAM3d(RAM3d):
    """ Grad RAM3d """

    def __init__(self, model, target_layer):
        super().__init__(model, target_layer)

        """
        Args:
            model: a base model to get RAM3d, which need not have global pooling and fully connected layer.
            target_layer: conv_layer you want to visualize
        """

    def forward(self, x):
        """
        Args:
            x: input image. shape =>(1, 3, H, W,t)
        Return:
            heatmap: class activation mappings of the predicted class
        """

        # anomaly detection
        score = self.model(x)

        # caluculate RAM3d of the predicted class
        RAM3d = self.getGradRAM3d(score)

        return RAM3d, score.cpu().data

    def __call__(self, x):
        return self.forward(x)

    def getGradRAM3d(self, score):
        '''
        values: the activations and gradients of target_layer
            activations: feature map before GAP.  shape => (1, C, H, W, T)
        score: the output of the model
        idx: predicted value
        RAM3d: class activation map.  shape=> (1, 1, H, W, T)
        '''

        self.model.zero_grad()

        score.backward(retain_graph=True)

        activations = self.values.activations
        gradients = self.values.gradients
        n, c, _, _, _ = gradients.shape

        alpha = gradients.view(n, c, -1).mean(2)
        alpha = alpha.view(n, c, 1, 1, 1)

        # shape => (1, 1, H', W, T')
        RAM3d = (alpha * activations).sum(dim=1, keepdim=True)
        RAM3d = F.relu(RAM3d)
        RAM3d -= torch.min(RAM3d)
        RAM3d /= torch.max(RAM3d)

        return RAM3d.data


class GradRAM3dpp(RAM3d):
    """ Grad RAM3d plus plus """

    def __init__(self, model, target_layer):
        super().__init__(model, target_layer)
        """
        Args:
            model: a base model
            target_layer: conv_layer you want to visualize
        """

    def forward(self, x):
        """
        Args:
            x: input image. shape =>(1, 3, H, W)
        Return:
            heatmap: class activation mappings of predicted classes
        """


        # anomaly detection
        score = self.model(x)

        # caluculate RAM3d of the predicted class
        RAM3d = self.getGradRAM3dpp(score)

        return RAM3d, score.cpu().data

    def __call__(self, x):
        return self.forward(x)

    def getGradRAM3dpp(self, score):
        '''
        values: the activations and gradients of target_layer
            activations: feature map before GAP.  shape => (1, C, H, W)
        score: the output of the model before softmax. shape => (1, n_classes)
        idx: predicted class id
        RAM3d: class activation map.  shape=> (1, 1, H, W)
        '''

        self.model.zero_grad()

        score.backward(retain_graph=True)

        activations = self.values.activations
        gradients = self.values.gradients
        n, c, _, _, _ = gradients.shape

        # calculate alpha
        numerator = gradients.pow(2)
        denominator = 2 * gradients.pow(2)
        ag = activations * gradients.pow(3)
        denominator += ag.view(n, c, -1).sum(-1, keepdim=True).view(n, c, 1, 1, 1)
        denominator = torch.where(
            denominator != 0.0, denominator, torch.ones_like(denominator))
        alpha = numerator / (denominator + 1e-7)

        relu_grad = F.relu(score.exp() * gradients)
        weights = (alpha * relu_grad).view(n, c, -1).sum(-1).view(n, c, 1, 1, 1)

        # shape => (1, 1, H', W')
        RAM3d = (weights * activations).sum(1, keepdim=True)
        RAM3d = F.relu(RAM3d)
        RAM3d -= torch.min(RAM3d)
        RAM3d /= torch.max(RAM3d)

        return RAM3d.data


class SmoothGradRAM3dpp(RAM3d):
    """ Smooth Grad RAM3d plus plus """

    def __init__(self, model, target_layer, n_samples=10, stdev_spread=0.15):
        super().__init__(model, target_layer)
        """
        Args:
            model: a base model
            target_layer: conv_layer you want to visualize
            n_sample: the number of samples
            stdev_spread: standard deviationß
        """

        self.n_samples = n_samples
        self.stdev_spread = stdev_spread

    def forward(self, x):
        """
        Args:
            x: input image. shape =>(1, 3, H, W)
        Return:
            heatmap: class activation mappings of predicted classes
        """

        stdev = self.stdev_spread / (x.max() - x.min())
        std_tensor = torch.ones_like(x) * stdev

        for i in range(self.n_samples):
            self.model.zero_grad()

            x_with_noise = torch.normal(mean=x, std=std_tensor)
            x_with_noise.requires_grad_()

            score = self.model(x_with_noise)

            score.backward(retain_graph=True)

            activations = self.values.activations
            gradients = self.values.gradients
            n, c, _, _,_ = gradients.shape

            # calculate alpha
            numerator = gradients.pow(2)
            denominator = 2 * gradients.pow(2)
            ag = activations * gradients.pow(3)
            denominator += \
                ag.view(n, c, -1).sum(-1, keepdim=True).view(n, c, 1, 1, 1)
            denominator = torch.where(
                denominator != 0.0, denominator, torch.ones_like(denominator))
            alpha = numerator / (denominator + 1e-7)

            relu_grad = F.relu(score.exp() * gradients)
            weights = \
                (alpha * relu_grad).view(n, c, -1).sum(-1).view(n, c, 1, 1, 1)

            # shape => (1, 1, H', W')
            RAM3d = (weights * activations).sum(1, keepdim=True)
            RAM3d = F.relu(RAM3d)
            RAM3d -= torch.min(RAM3d)
            RAM3d /= torch.max(RAM3d)

            if i == 0:
                total_RAM3ds = RAM3d.clone()
            else:
                total_RAM3ds += RAM3d

        total_RAM3ds /= self.n_samples

        return total_RAM3ds.data, score

    def __call__(self, x):
        return self.forward(x)
