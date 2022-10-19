import torch
import torch.nn as nn


class CenterLoss(nn.Module):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=10, feat_dim=2, device=None, constrained_classes=None, mu=0.5):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device

        # random initialization of the centers. Will be later corrected based on input data
        centers = torch.randn(self.num_classes, self.feat_dim)
        if device:
            self.centers = centers.to(device)
        else:
            self.centers = centers

        self.centers_were_set = False
        self.mu = mu

        self.constrained_classes = constrained_classes

        # if constrained_classes is not None:
        #     print('center loss is used for class {} only'.format(constrained_classes))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """

        # update centers based on input x
        with torch.no_grad():
            for label in range(self.num_classes):
                idx = labels == label
                class_available = torch.any(idx)

                if class_available:
                    new_center = x[idx].mean(dim=0)

                    if self.centers_were_set:
                        # this corresponds to the paper gradient update for centers, but here I use explicite update without the need of pytorch learning the centers
                        self.centers[label] = (1 - self.mu) * self.centers[label] + self.mu * new_center
                    else:
                        self.centers[label] = new_center
                        self.centers_were_set = True

        batch_size = x.size(0)

        # ||c||**2 + ||I||**2-2C*I
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(x, self.centers.t(), beta=1, alpha=-2)

        # simplier way to compute distances - already existing in pytorch (includes unecessary square root)
        # But there is some problem with Nans, probably after taking a root from negative, at list at 1.2 version of pytorch
        # distmat = torch.cdist(x, self.centers)**2

        classes = torch.arange(self.num_classes).long()
        if self.device: classes = classes.to(self.device)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        n_per_class = mask.sum(dim=0)
        n_per_class[n_per_class == 0] = 1 # to prevent dividing by 0

        dist = distmat * mask.float()
        dist = dist.clamp(min=1e-12, max=1e+12)
        loss = dist.sum(dim=0)
        loss = loss / n_per_class.float()

        if self.constrained_classes:
            loss_within = loss[self.constrained_classes]
        else:
            #loss_within = loss.sum()
            loss_within = loss

        loss_within = loss_within.mean()

        # # # between class centers loss
        # mask = ~mask
        # n_per_class = mask.sum(dim=0)
        # n_per_class[n_per_class == 0] = 1  # to prevent dividing by 0
        #
        # dist = distmat * mask.float()
        # dist = dist.clamp(min=1e-12, max=1e+12)
        # loss = dist.sum(dim=0)
        # loss = loss / n_per_class.float()
        #
        # if self.constrained_classes:
        #     loss_between = loss[self.constrained_classes]
        # else:
        #     # loss_between = loss.sum()
        #     loss_between = loss
        #
        # loss_between = loss_between.mean()

        #loss = loss_within / loss_between
        #loss = loss_within / math.sqrt(loss_between)
        #loss = loss_within - 0.2*loss_between
        loss = loss_within

        loss = loss/x.size(1) # making centerloss independent from the dimensionality of feature vectors (not needed when using ratio of distances)
        #loss = math.sqrt(loss)
        return loss

class CenterLossOld(nn.Module):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=10, feat_dim=2, device=None, one_class=None):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim

        if device is None:
            self.device = None
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

        else:
            self.device = device
            self.centers = nn.Parameter((torch.randn(self.num_classes, self.feat_dim)).to(self.device))

        self.one_class = one_class
        if one_class is not None:
            print('center loss is used for class {} only'.format(one_class))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        #distmat.addmm_(1, -2, x, self.centers.t())
        distmat.addmm_(x, self.centers.t(), beta=1, alpha=-2)

        classes = torch.arange(self.num_classes).long()
        if self.device: classes = classes.to(self.device)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        dist = dist.clamp(min=1e-12, max=1e+12)
        loss = dist.sum(dim=0)

        if self.one_class:
            loss = loss[self.one_class]
        else:
            loss = loss.sum()

        loss = loss / batch_size

        return loss


