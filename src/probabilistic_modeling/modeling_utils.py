import copy
import math

import torch
from sklearn.mixture._gaussian_mixture import _compute_precision_cholesky
from torch import nn
from torch.distributions import Distribution
from torch.distributions.categorical import Categorical
from torch.distributions.independent import Independent
from torch.distributions.laplace import Laplace
from torch.distributions.mixture_same_family import MixtureSameFamily
from torch.distributions.multivariate_normal import MultivariateNormal


class ClassRegDist(Distribution):
    def __init__(
        self,
        loc,
        reg_dist,
        reg_kwargs,
        probs=None,
        logits=None,
        independent_reg_dist=False,
    ):
        batch_shape = loc.shape[:-1]
        event_shape = torch.Size([1 + loc.shape[-1]])

        self.reg_dist = reg_dist(loc, **reg_kwargs)
        if independent_reg_dist:
            self.reg_dist = Independent(self.reg_dist, 1)
        self.cls_dist = Categorical(probs=probs, logits=logits)

        self.dist_type = "log_prob"

        super().__init__(batch_shape, event_shape, validate_args=False)

    def log_prob(self, value):
        
        cls_log_prob = self.cls_dist.log_prob(value[..., -1])

        if self.dist_type == "euclidian":
            reg_log_prob = -(self.reg_dist.mean - value[..., :-1]).pow(2).sum(-1).sqrt()
        elif self.dist_type == "euclidian_squared":
            reg_log_prob = -(self.reg_dist.mean - value[..., :-1]).pow(2).sum(-1)
        else:
            reg_log_prob = self.reg_dist.log_prob(value[..., :-1])
        return cls_log_prob + reg_log_prob

    def set_dist_mode(self, dist_type):
        self.dist_type = dist_type


def unscented_transform(means, chols, anchors, trans_func):
    """ Definition 1 in https://arxiv.org/abs/2104.01958

    Args:
        means (_type_): _description_
        chols (_type_): _description_
        anchors (_type_): _description_
        trans_func (_type_): _description_

    Returns:
        _type_: _description_
    """
    n = means.shape[-1]
    kappa = n-3
    if len(means.shape) > 2:
        old_means_shape = means.shape
        means = means.reshape(-1,n)
    if len(chols > 3):
        old_chol_shape = chols.shape
        chols = chols.reshape(-1,n,n)
    
    N = len(means)
    weights = torch.ones((1,2*n+1,1), device=means.device)/(2*(n+kappa))
    weights[0,0,0] = kappa / (n+kappa)
    # means [N, n], chols [N, n, n]
    # [N, 1, n]
    sigma_points1 = means.unsqueeze(1)
    # [N, n, n]
    sigma_points2 = means.unsqueeze(1) + math.sqrt(n+kappa)*chols
    # [N, n, n]
    sigma_points3 = means.unsqueeze(1) - math.sqrt(n+kappa)*chols
    # [N, 2n+1, n]
    sigma_points = torch.cat((sigma_points1, sigma_points2, sigma_points3), dim=1)

    repeated_anchors = anchors.repeat_interleave(len(means)//len(anchors),dim=0).unsqueeze(1).repeat(1,2*n+1,1).reshape(-1,n)

    transformed_sigma_points = trans_func(sigma_points.reshape(-1, n), repeated_anchors)
    transformed_sigma_points = transformed_sigma_points.reshape(N, 2*n+1, n)

    transformed_means = (transformed_sigma_points*weights).sum(dim=1)
    residuals = transformed_sigma_points-transformed_means.unsqueeze(1)
    # [N, 2n+1, n, 1]
    residuals = residuals.unsqueeze(-1)
    # [N, n, n]
    transformed_covs = (weights.unsqueeze(-1)*residuals@residuals.transpose(-1,-2)).sum(dim=1)

    transformed_chols, info = torch.linalg.cholesky_ex(transformed_covs)
    if not (info==0).all():
        # Clamp to avoid errors
        transformed_chols = torch.diag_embed(torch.diagonal(transformed_chols,dim1=-2,dim2=-1).clamp(math.exp(-7),math.exp(10)))+torch.tril(transformed_chols,-1)
        print("***************************")
        for cov,res,trans_mean,mean,anchor,chol in zip(transformed_covs[info!=0], residuals[info!=0].squeeze(-1), transformed_means[info!=0], means[info!=0], anchors.repeat_interleave(len(means)//len(anchors),dim=0)[info!=0], chols[info!=0]):
            print(cov)
            print(res)
            print(trans_mean)
            print(mean)
            print(anchor)
            print(chol)
            print("+++++++++++++++++++++++++++++++++++")
        
        print("***************************")
    
    return transformed_means.reshape(old_means_shape), transformed_chols.reshape(old_chol_shape)


def covariance_output_to_cholesky(pred_bbox_cov):
    """
    Transforms output to covariance cholesky decomposition.
    Args:
        pred_bbox_cov (kx4 or kx10): Output covariance matrix elements.

    Returns:
        predicted_cov_cholesky (kx4x4): cholesky factor matrix
    """
    # Embed diagonal variance
    if pred_bbox_cov.shape[0] == 0:
        return pred_bbox_cov.reshape((0, 4, 4))

    diag_vars = torch.sqrt(torch.exp(pred_bbox_cov[..., :4]))
    predicted_cov_cholesky = torch.diag_embed(diag_vars)

    if pred_bbox_cov.shape[-1] > 4:
        tril_indices = torch.tril_indices(row=4, col=4, offset=-1)
        predicted_cov_cholesky[..., tril_indices[0], tril_indices[1]] = pred_bbox_cov[
            ..., 4:
        ]

    return predicted_cov_cholesky


def clamp_log_variance(pred_bbox_cov, clamp_min=-7.0, clamp_max=10.0):
    """
    Tiny function that clamps variance for consistency across all methods.
    """
    pred_bbox_var_component = torch.clamp(pred_bbox_cov[..., 0:4], clamp_min, clamp_max)
    return torch.cat((pred_bbox_var_component, pred_bbox_cov[..., 4:]), dim=-1)


def get_probabilistic_loss_weight(current_step, annealing_step):
    """
    Tiny function to get adaptive probabilistic loss weight for consistency across all methods.
    """
    probabilistic_loss_weight = min(1.0, current_step / annealing_step)
    probabilistic_loss_weight = (100 ** probabilistic_loss_weight - 1.0) / (100.0 - 1.0)

    return probabilistic_loss_weight


def freeze_non_probabilistic_weights(cfg, model):
    """
    Tiny function to only keep a small subset of weight non-frozen.
    """
    if cfg.MODEL.TRAIN_ONLY_PPP:
        print("[NLLOD]: Freezing all non-PPP weights")
        for name, p in model.named_parameters():
            if "ppp_intensity_function" in name:
                p.requires_grad = cfg.MODEL.TRAIN_PPP
            else:
                p.requires_grad = False
        print("[NLLOD]: Froze all non-PPP weights")

    elif cfg.MODEL.TRAIN_ONLY_UNCERTAINTY_PREDS:
        print("[NLLOD]: Freezing all non-probabilistic weights")
        for name, p in model.named_parameters():
            if "ppp_intensity_function" in name:
                p.requires_grad = cfg.MODEL.TRAIN_PPP
            elif "bbox_cov" in name:
                p.requires_grad = True
            else:
                p.requires_grad = False
        print("[NLLOD]: Froze all non-probabilistic weights")

    else:
        for name, p in model.named_parameters():
            if "ppp_intensity_function" in name:
                p.requires_grad = cfg.MODEL.TRAIN_PPP


class PoissonPointProcessBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.normalize_bboxes = False

    def set_normalization_of_bboxes(self, normalize_bboxes):
        self.normalize_bboxes = normalize_bboxes


class PoissonPointUnion(PoissonPointProcessBase):
    def __init__(self):
        super().__init__()
        self.ppps = []

    def add_ppp(self, ppp):
        self.ppps.append(ppp)

    def set_normalization_of_bboxes(self, normalize_bboxes):
        for ppp in self.ppps:
            ppp.normalize_bboxes = normalize_bboxes

    def integrate(self, image_sizes, num_classes):
        out = 0
        for ppp in self.ppps:
            out = out + ppp.integrate(image_sizes, num_classes)
        return out 

    def forward(
        self,
        src,
        image_sizes=[],
        num_classes=-1,
        integrate=False,
        src_is_features=False,
        dist_type="log_prob",
    ):
        if integrate:
            out = self.integrate(image_sizes, num_classes)
            return out

        outs = []
        for ppp in self.ppps:
            outs.append(
                ppp(src, image_sizes, num_classes, integrate, src_is_features, dist_type)[:, None]
            )
        outs = torch.cat(outs, 1)
        return torch.logsumexp(outs, 1)

class PoissonPointProcessUniform(PoissonPointProcessBase):
    def __init__(
        self,
        class_dist_log,
        ppp_rate,
        uniform_center_pos,
        device=torch.device("cpu"),
    ):
        super().__init__()
        if not type(class_dist_log) == torch.Tensor:
            class_dist_log = torch.tensor(class_dist_log)

        self.class_dist_log = class_dist_log.to(device)
        self.ppp_rate = torch.tensor([ppp_rate]).to(device)
        self.uniform_center_pos = uniform_center_pos
        self.device = device

    def forward(
        self,
        src,
        image_sizes=[],
        num_classes=-1,
        integrate=False,
        src_is_features=False,
    ):
        if integrate:
            return self.integrate(image_sizes, num_classes)

        assert len(image_sizes) == 1
        img_size = image_sizes[0].flip(0).repeat(2)  # w,h,w,h

        cls_log_probs = self.class_dist_log[src[..., -1].long()]
        # log(1/(W^2/2) * 1/(H^2/2))
        box_log_probs = (-image_sizes[0].log()*2+math.log(2)).sum()

        total_log_probs = cls_log_probs + box_log_probs + self.ppp_rate.log()

        return total_log_probs

    def integrate(self, image_sizes, num_classes):
        return self.ppp_rate

class PoissonPointProcessGMM(PoissonPointProcessBase):
    def __init__(
        self,
        gmm,
        class_dist_log,
        ppp_rate,
        uniform_center_pos,
        device=torch.device("cpu"),
    ):
        super().__init__()
        if not type(class_dist_log) == torch.Tensor:
            class_dist_log = torch.tensor(class_dist_log)

        self.class_dist_log = class_dist_log.to(device)
        self.gmm = gmm
        self.ppp_rate = torch.tensor([ppp_rate]).to(device)
        self.uniform_center_pos = uniform_center_pos
        self.device = device

    def forward(
        self,
        src,
        image_sizes=[],
        num_classes=-1,
        integrate=False,
        src_is_features=False,
    ):
        if integrate:
            return self.integrate(image_sizes, num_classes)

        assert len(image_sizes) == 1
        img_size = image_sizes[0].flip(0).repeat(2)  # w,h,w,h
        scale = torch.diag_embed(img_size).cpu().numpy()
        gmm = copy.deepcopy(self.gmm)

        boxes = src[..., :-1]
        if self.uniform_center_pos:
            gmm.means_ = gmm.means_ * img_size.cpu().numpy()[:2]
            gmm.covariances_ = scale[:2, :2] @ gmm.covariances_ @ scale[:2, :2].T
            gmm.precisions_cholesky_ = _compute_precision_cholesky(
                gmm.covariances_, gmm.covariance_type
            )

            img_area = img_size[0] * img_size[1]
            # N, 2 (w,h)
            box_sizes = torch.cat(
                (
                    (boxes[..., 2] - boxes[..., 0])[:, None],
                    (boxes[..., 3] - boxes[..., 1])[:, None],
                ),
                1,
            )
            box_log_probs = torch.tensor(gmm.score_samples(box_sizes.cpu().numpy())).to(
                box_sizes.device
            )
            box_log_probs = box_log_probs - img_area.log()

        else:
            gmm.means_ = gmm.means_ * img_size.cpu().numpy()
            gmm.covariances_ = scale @ gmm.covariances_ @ scale.T
            gmm.precisions_cholesky_ = _compute_precision_cholesky(
                gmm.covariances_, gmm.covariance_type
            )
            box_log_probs = torch.tensor(gmm.score_samples(boxes.cpu().numpy())).to(
                boxes.device
            )

        cls_log_probs = self.class_dist_log[src[..., -1].long()]

        total_log_probs = cls_log_probs + box_log_probs + self.ppp_rate.log()

        return total_log_probs

    def integrate(self, image_sizes, num_classes):
        return self.ppp_rate

class ZeroDistribution(PoissonPointProcessBase):
    def __init__(self, device=torch.device("cuda"))-> None:
        super().__init__()
        self.device = device
        self.component_distribution = None

    def log_prob(self, src, *args, **kwargs):
        return torch.tensor(0.0).to(src.device).unsqueeze(0).repeat(len(src)).log()

class PoissonPointProcessIntensityFunction(PoissonPointProcessBase):
    """
    Class representing a Poisson Point Process RFS intensity function. Currently assuming DETR/RCNN/RetinaNet.
    """

    def __init__(
        self, cfg, log_intensity=None, ppp_feature_net=None, predictions=None, device="cuda"
    ) -> None:
        super().__init__()
        self.device = device
        if cfg.PROBABILISTIC_INFERENCE.PPP_CONFIDENCE_THRES and predictions is not None:
            self.ppp_intensity_type = "prediction_mixture"
        elif log_intensity is not None:
            self.ppp_intensity_type = "uniform"
            self.num_classes = 1
        else:
            self.ppp_intensity_type = (
                cfg.MODEL.PROBABILISTIC_MODELING.PPP.INTENSITY_TYPE
            )
            self.num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES

        self.ppp_confidence_thres = cfg.PROBABILISTIC_INFERENCE.PPP_CONFIDENCE_THRES
        self.ppp_feature_net = ppp_feature_net

        if self.ppp_intensity_type == "uniform":
            self.ppp_intensity_per_coord = nn.Parameter(
                torch.tensor(1.0).to(self.device), requires_grad=True
            )
            self.log_ppp_intensity_class = nn.Parameter(
                torch.tensor(1.0).to(self.device), requires_grad=True
            )

            if log_intensity is None:
                nn.init.constant_(
                    self.ppp_intensity_per_coord,
                    cfg.MODEL.PROBABILISTIC_MODELING.PPP.UNIFORM_INTENSITY,
                )
                nn.init.constant_(
                    self.log_ppp_intensity_class,
                    math.log(1 / cfg.MODEL.ROI_HEADS.NUM_CLASSES),
                )
            else:
                nn.init.constant_(self.ppp_intensity_per_coord, log_intensity)
                nn.init.constant_(self.log_ppp_intensity_class, 0)
                self.log_ppp_intensity_class.requires_grad = False

        elif self.ppp_intensity_type == "gaussian_mixture":
            num_mixture_comps = cfg.MODEL.PROBABILISTIC_MODELING.PPP.NUM_GAUSS_MIXTURES
            cov_type = cfg.MODEL.PROBABILISTIC_MODELING.PPP.COV_TYPE
            if cov_type == "diagonal":
                cov_dims = 4
            elif cov_type == "full":
                cov_dims = 10
            else:
                cov_dims = 4

            self.log_gmm_weights = nn.Parameter(
                (torch.ones(num_mixture_comps)*0.5).log().to(self.device),
                requires_grad=True,
            )
            nn.init.normal_(self.log_gmm_weights, mean=0, std=0.1)

            means = torch.distributions.Normal(torch.tensor([0.5]).to(self.device), scale=torch.tensor([0.16]).to(self.device)).rsample((num_mixture_comps, 4,)).squeeze(-1)
            xywh_to_xyxy = torch.tensor([[1,0,-0.5,0],[0,1,0,-0.5],[1,0,0.5,0],[0,1,0,0.5]]).to(self.device)
            means = (xywh_to_xyxy@(means.unsqueeze(-1))).squeeze(-1)
            means = means.clamp(0,1)
            self.gmm_means = nn.Parameter(
                means, requires_grad=True
            )
            self.gmm_chols = nn.Parameter(
                torch.zeros(num_mixture_comps, cov_dims).to(self.device), requires_grad=True
            )
            nn.init.normal_(self.gmm_chols, std=1)
            
            cls_probs = torch.ones(num_mixture_comps, self.num_classes).to(self.device)/self.num_classes + torch.rand((num_mixture_comps, self.num_classes)).to(self.device)*0.1
            cls_logits = (cls_probs/(1-cls_probs)).log()
            self.class_logits = nn.Parameter(
                cls_logits, requires_grad=True
            )  # these are softmaxed later

            #self.mvn = MultivariateNormal(self.gmm_means, scale_tril=self.gmm_chols)

            reg_kwargs = {"scale_tril": covariance_output_to_cholesky(self.gmm_chols)}

            mixture_dict = {}
            mixture_dict["means"] = self.gmm_means
            mixture_dict["weights"] = self.log_gmm_weights.exp()
            mixture_dict["reg_dist"] = torch.distributions.multivariate_normal.MultivariateNormal
            mixture_dict["reg_kwargs"] = reg_kwargs
            mixture_dict["cls_probs"] = self.class_logits.softmax(dim=-1)
            mixture_dict["reg_dist_type"] = "gaussian"
            mixture_dict["covs"] = None

            self.mixture_from_predictions(mixture_dict)

        elif self.ppp_intensity_type == "prediction_mixture":
            if predictions is not None:
                self.mixture_from_predictions(predictions)

        elif self.ppp_intensity_type == "zero":
            self.dist = ZeroDistribution(self.device)
        else:
            raise NotImplementedError(
                f"PPP intensity type {cfg.MODEL.PROBABILISTIC_MODELING.PPP_INTENSITY_TYPE} not implemented."
            )

    def mixture_from_predictions(self, mixture_dict):
        reg_dist_str = mixture_dict["reg_dist_type"]
        means = mixture_dict["means"]
        covs = mixture_dict["covs"]
        weights = mixture_dict["weights"]
        cls_probs = mixture_dict["cls_probs"]

        reg_kwargs = mixture_dict["reg_kwargs"]
        independent_reg_dist = False
        reg_dist = mixture_dict["reg_dist"]
        if reg_dist_str == "laplacian":
            independent_reg_dist = True
        if not len(weights):
            self.mixture_dist = ZeroDistribution(means.device)
            self.ppp_rate = torch.tensor(0.0).to(means.device)
        else:
            self.mixture_dist = MixtureSameFamily(
                Categorical(weights),
                ClassRegDist(
                    means,
                    reg_dist,
                    reg_kwargs,
                    probs=cls_probs,
                    independent_reg_dist=independent_reg_dist,
                ),
                validate_args=False,
            )
            self.ppp_rate = weights.sum()

    def get_weights(self):
        weights = dict()
        if self.ppp_intensity_type == "uniform":
            weights["ppp_intensity_per_coord"] = self.ppp_intensity_per_coord
            weights["log_ppp_intensity_class"] = self.log_ppp_intensity_class

        elif self.ppp_intensity_type == "gaussian_mixture":
            return weights
            weights["log_gmm_weights"] = self.log_gmm_weights
            weights["gmm_means"] = self.gmm_means
            weights["gmm_covs"] = self.gmm_covs
            weights["class_weights"] = self.class_weights
            weights["log_class_scaling"] = self.log_class_scaling

        return weights

    def load_weights(self, weights):
        if self.ppp_intensity_type == "uniform":
            self.ppp_intensity_per_coord = nn.Parameter(
                torch.as_tensor(weights["ppp_intensity_per_coord"])
            )
            self.log_ppp_intensity_class = nn.Parameter(
                torch.as_tensor(weights["log_ppp_intensity_class"])
            )

        elif self.ppp_intensity_type == "gaussian_mixture":
            self.log_gmm_weights = nn.Parameter(
                torch.as_tensor(weights["log_gmm_weights"])
            )
            self.gmm_means = nn.Parameter(torch.as_tensor(weights["gmm_means"]))
            self.gmm_covs = nn.Parameter(torch.as_tensor(weights["gmm_covs"]))
            self.class_weights = nn.Parameter(torch.as_tensor(weights["class_weights"]))
            self.log_class_scaling = nn.Parameter(
                torch.as_tensor(weights["log_class_scaling"])
            )
            self.update_distribution()

    def update_distribution(self):
        if self.ppp_intensity_type == "gaussian_mixture":
            mixture_dict = {}
            mixture_dict["means"] = self.gmm_means
            mixture_dict["weights"] = self.log_gmm_weights.exp()
            mixture_dict["reg_dist"] = torch.distributions.multivariate_normal.MultivariateNormal
            mixture_dict["reg_kwargs"] = {"scale_tril": covariance_output_to_cholesky(self.gmm_chols)}
            mixture_dict["cls_probs"] = self.class_logits.softmax(dim=-1)
            mixture_dict["reg_dist_type"] = "gaussian"
            mixture_dict["covs"] = None

            self.mixture_from_predictions(mixture_dict)

    def forward_features(self, src):
        print("[NLLOD] Data dependent PPP not available yet")
        return

        out = self.ppp_feature_net(src)
        if self.ppp_intensity_type == "gaussian_mixture":
            pass
            # translate output to gmm params
        return

    def forward(
        self,
        src,
        image_sizes=[],
        num_classes=-1,
        integrate=False,
        src_is_features=False,
        dist_type="log_prob"
    ):
        """Calculate log PPP intensity for given input. If numclasses =! -1, returns integral over intensity

        Args:
            src ([type]): [description]
            image_sizes (list, optional): [description]. Defaults to [].
            num_classes (int, optional): [description]. Defaults to -1.

        Returns:
            [type]: [description]
        """
        if src_is_features:
            return self.forward_features(src)

        if integrate:
            return self.integrate(image_sizes, num_classes)

        if self.ppp_intensity_type == "uniform":
            # Returns log intensity func value
            coord_log_prob = self.ppp_intensity_per_coord
            if src.shape[-1] > 4:
                src = src[..., :4]
            # keep gradients trough src, +1 to handle coodinates in zero
            out = (src + 1) / (src.detach() + 1) * coord_log_prob
            out = out.sum(-1)
            class_log_prob = self.log_ppp_intensity_class
            out = out + class_log_prob
        elif self.ppp_intensity_type == "gaussian_mixture":
            if self.normalize_bboxes:
                # H,W -> (flip) -> W,H -> (repeat) -> W,H,W,H
                box_scaling = 1/image_sizes.flip((-1)).repeat(1,2).float()
                class_scaling = torch.ones((len(image_sizes),1)).to(src.device)
                # [1, 5]
                scaling = torch.cat([box_scaling, class_scaling], dim=-1)
                # [num_gt, 5]
                scaling = scaling.repeat(src.shape[0],1)
                src = src*scaling
            else:
                scaling = torch.ones_like(src)

            if self.mixture_dist.component_distribution:
                self.mixture_dist.component_distribution.set_dist_mode(dist_type)
            out = self.mixture_dist.log_prob(src)
            out = out + self.ppp_rate.log()

            out = out + scaling.log().sum(dim=-1)

        elif self.ppp_intensity_type == "prediction_mixture":
            if self.mixture_dist.component_distribution:
                self.mixture_dist.component_distribution.set_dist_mode(dist_type)
            out = self.mixture_dist.log_prob(src)
            out = out + self.ppp_rate.log()
        elif self.ppp_intensity_type == "zero":
            out = self.dist.log_prob(src)

        return out

    def integrate(self, image_sizes, num_classes):
        if self.ppp_intensity_type == "uniform":
            # Evaluate the integral of the intensity funciton of all possible inputs
            coord_log_prob = self.ppp_intensity_per_coord
            class_log_prob = self.log_ppp_intensity_class
            # Divide by 2 because x1 < x2 and y1 < y2
            image_part = torch.log(
                image_sizes[:, 0] ** 2 / 2 * image_sizes[:, 1] ** 2 / 2
            ) + (4 * coord_log_prob)
            class_part = math.log(num_classes) + class_log_prob
            out = (image_part + class_part).exp()
        elif self.ppp_intensity_type == "gaussian_mixture":
            out = self.ppp_rate
        elif self.ppp_intensity_type == "prediction_mixture":
            out = self.ppp_rate
        elif self.ppp_intensity_type == "zero":
            out = torch.zeros(len(image_sizes)).to(image_sizes.device)
        else:
            out = torch.zeros(len(image_sizes)).to(image_sizes.device)

        return out
