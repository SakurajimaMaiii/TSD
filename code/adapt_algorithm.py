import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F



@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def collect_params(model):
    """Collect the affine scale + shift parameters from batch norms.
    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.
    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what tent updates
    model.requires_grad_(False)
    # configure norm for tent updates: enable grad + force batch statisics
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
    return model


def check_model(model):
    """Check model for compatability with tent."""
    is_training = model.training
    assert is_training, "tent needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "tent needs params to update: " \
                           "check which require grad"
    assert not has_all_params, "tent should not update all params: " \
                               "check which require grad"
    has_bn = any([isinstance(m, nn.BatchNorm2d) for m in model.modules()])
    assert has_bn, "tent needs normalization for its optimization"


class ERM(nn.Module):
    def __init__(self,model):
        super().__init__()
        self.model = model.eval()
        self.featurizer = model.featurizer
        self.classifier = model.classifier
        
    @torch.no_grad()
    def forward(self, x):
        outputs = self.model.predict(x)
        return outputs
        

class BN(nn.Module):
    def __init__(self, model, steps=1, episodic=False):
        super().__init__()
        self.model = model
        self.steps = steps
        self.episodic = episodic
        assert self.steps>=0, 'steps must be non-negative'
        if self.steps==0:
            self.model.eval()
    
    @torch.no_grad()
    def forward(self, x):
        if self.steps>0:
            for _ in range(self.steps):
                outputs = self.model.predict(x)
        else:
            outputs = self.model.predict(x)
        return outputs


class Tent(nn.Module):
    """
    ICLR,2021
    Tent adapts a model by entropy minimization during testing.
    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, optimizer, steps=1, episodic=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.episodic = episodic

        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            outputs = self.forward_and_adapt(x, self.model, self.optimizer)

        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)
    
    @torch.enable_grad()
    def forward_and_adapt(self,x, model, optimizer):
        """Forward and adapt model on batch of data.
        Measure entropy of the model prediction, take gradients, and update params.
        """
        # forward
        outputs = model.predict(x)
        # adapt
        loss = softmax_entropy(outputs).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()        
        return outputs


class PseudoLabel(nn.Module):
    def __init__(self, model, optimizer, beta=0.9,steps=1, episodic=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "requires >= 1 step(s) to forward and update"
        self.episodic = episodic
        self.beta = beta  #threshold for selecting pseudo labels
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            outputs = self.forward_and_adapt(x, self.model, self.optimizer)

        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)
    
    @torch.enable_grad()
    def forward_and_adapt(self,x, model, optimizer):
        # forward
        outputs = model.predict(x)
        # adapt        
        scores = F.softmax(outputs,1)
        py,y_prime = torch.max(scores,1)
        mask = py > self.beta
        loss = F.cross_entropy(outputs[mask],y_prime[mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return outputs


class SHOTIM(nn.Module):
    """
    SHOT-IM ,ICML 2020
    """
    def __init__(self, model, optimizer, steps=1, episodic=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "requires >= 1 step(s) to forward and update"
        self.episodic = episodic

        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            outputs = self.forward_and_adapt(x, self.model, self.optimizer)

        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)
    
    @torch.enable_grad()
    def forward_and_adapt(self,x, model, optimizer):
        # forward
        outputs = model.predict(x)
        # adapt
        ent_loss = softmax_entropy(outputs).mean(0)
        softmax_out = F.softmax(outputs, dim=1)
        msoftmax = softmax_out.mean(dim=0)
        div_loss = torch.sum(msoftmax * torch.log(msoftmax + 1e-5))
        loss = ent_loss + div_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return outputs
        
        
class T3A(nn.Module):
    """
    T3A, NeurIPS 2021
    """
    def __init__(self,model,filter_K=100,steps=1,episodic=False):
        super().__init__()
        self.model = model.eval()
        self.featurizer = model.featurizer
        self.classifier = model.classifier
        self.steps = steps
        assert steps > 0, "requires >= 1 step(s) to forward and update"
        self.episodic = episodic
        self.filter_K = filter_K
        
        warmup_supports = self.classifier.fc.weight.data
        self.num_classes = warmup_supports.size()[0]
        self.warmup_supports = warmup_supports
        warmup_prob = self.classifier(self.warmup_supports)
        self.warmup_ent = softmax_entropy(warmup_prob)
        self.warmup_labels = F.one_hot(warmup_prob.argmax(1), num_classes=self.num_classes).float()
        self.supports = self.warmup_supports.data
        self.labels = self.warmup_labels.data
        self.ent = self.warmup_ent.data
             
    @torch.no_grad() 
    def forward(self,x):
        z = self.featurizer(x)
        p = self.classifier(z)
        yhat = torch.nn.functional.one_hot(p.argmax(1), num_classes=self.num_classes).float()
        ent = softmax_entropy(p)

        self.supports = self.supports.to(z.device)
        self.labels = self.labels.to(z.device)
        self.ent = self.ent.to(z.device)
        self.supports = torch.cat([self.supports,z])
        self.labels = torch.cat([self.labels,yhat])
        self.ent = torch.cat([self.ent,ent])
        
        supports, labels = self.select_supports()
        supports = torch.nn.functional.normalize(supports, dim=1)
        weights = (supports.T @ (labels))
        
        return z @ torch.nn.functional.normalize(weights, dim=0)

    def select_supports(self):
        ent_s = self.ent
        y_hat = self.labels.argmax(dim=1).long()
        filter_K = self.filter_K
        if filter_K == -1:
            indices = torch.LongTensor(list(range(len(ent_s))))

        indices = []
        indices1 = torch.LongTensor(list(range(len(ent_s)))).cuda()
        for i in range(self.num_classes):
            _, indices2 = torch.sort(ent_s[y_hat == i])
            indices.append(indices1[y_hat==i][indices2][:filter_K])
        indices = torch.cat(indices)

        self.supports = self.supports[indices]
        self.labels = self.labels[indices]
        self.ent = self.ent[indices]
        
        return self.supports, self.labels
        

###################################
class TSD(nn.Module):
    """
    Test-time Self-Distillation (TSD)
    CVPR 2023
    """
    def __init__(self,model,optimizer,lam=0,filter_K=100,steps=1,episodic=False):
        super().__init__()
        self.model = model
        self.featurizer = model.featurizer
        self.classifier = model.classifier
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "requires >= 1 step(s) to forward and update"
        self.episodic = episodic
        self.filter_K = filter_K
        
        warmup_supports = self.classifier.fc.weight.data.detach()
        self.num_classes = warmup_supports.size()[0]
        self.warmup_supports = warmup_supports
        warmup_prob = self.classifier(self.warmup_supports)
        
        self.warmup_ent = softmax_entropy(warmup_prob)
        self.warmup_labels = F.one_hot(warmup_prob.argmax(1), num_classes=self.num_classes).float()
        self.warmup_scores = F.softmax(warmup_prob,1)
                
        self.supports = self.warmup_supports.data
        self.labels = self.warmup_labels.data
        self.ent = self.warmup_ent.data
        self.scores = self.warmup_scores.data
        self.lam = lam
        
        
    def forward(self,x):
        z = self.featurizer(x)
        p = self.classifier(z)
                       
        yhat = F.one_hot(p.argmax(1), num_classes=self.num_classes).float()
        ent = softmax_entropy(p)
        scores = F.softmax(p,1)

        with torch.no_grad():
            self.supports = self.supports.to(z.device)
            self.labels = self.labels.to(z.device)
            self.ent = self.ent.to(z.device)
            self.scores = self.scores.to(z.device)
            self.supports = torch.cat([self.supports,z])
            self.labels = torch.cat([self.labels,yhat])
            self.ent = torch.cat([self.ent,ent])
            self.scores = torch.cat([self.scores,scores])
        
            supports, labels = self.select_supports()
            supports = F.normalize(supports, dim=1)
            weights = (supports.T @ (labels))
                
        dist,loss = self.prototype_loss(z,weights.T,scores,use_hard=False)

        loss_local = topk_cluster(z.detach().clone(),supports,self.scores,p,k=3)
        loss += self.lam*loss_local
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return p

    def select_supports(self):
        ent_s = self.ent
        y_hat = self.labels.argmax(dim=1).long()
        filter_K = self.filter_K
        if filter_K == -1:
            indices = torch.LongTensor(list(range(len(ent_s))))

        indices = []
        indices1 = torch.LongTensor(list(range(len(ent_s)))).cuda()
        for i in range(self.num_classes):
            _, indices2 = torch.sort(ent_s[y_hat == i])
            indices.append(indices1[y_hat==i][indices2][:filter_K])
        indices = torch.cat(indices)

        self.supports = self.supports[indices]
        self.labels = self.labels[indices]
        self.ent = self.ent[indices]
        self.scores = self.scores[indices]
        
        return self.supports, self.labels
    
    def prototype_loss(self,z,p,labels=None,use_hard=False,tau=1):
        #z [batch_size,feature_dim]
        #p [num_class,feature_dim]
        #labels [batch_size,]        
        z = F.normalize(z,1)
        p = F.normalize(p,1)
        dist = z @ p.T / tau
        if labels is None:
            _,labels = dist.max(1)
        if use_hard:
            """use hard label for supervision """
            #_,labels = dist.max(1)  #for prototype-based pseudo-label
            labels = labels.argmax(1)  #for logits-based pseudo-label
            loss =  F.cross_entropy(dist,labels)
        else:
            """use soft label for supervision """
            loss = softmax_kl_loss(labels.detach(),dist).sum(1).mean(0)  #detach is **necessary**
            #loss = softmax_kl_loss(dist,labels.detach()).sum(1).mean(0) achieves comparable results
        return dist,loss
        

def topk_labels(feature,supports,scores,k=3):
    feature = F.normalize(feature,1)
    supports = F.normalize(supports,1)
    sim_matrix = feature @ supports.T  #B,M
    _,idx_near = torch.topk(sim_matrix,k,dim=1)  #batch x K
    scores_near = scores[idx_near]  #batch x K x num_class
    soft_labels = torch.mean(scores_near,1)  #batch x num_class
    soft_labels = torch.argmax(soft_labels,1)
    return soft_labels
    

def topk_cluster(feature,supports,scores,p,k=3):
    #p: outputs of model batch x num_class
    feature = F.normalize(feature,1)
    supports = F.normalize(supports,1)
    sim_matrix = feature @ supports.T  #B,M
    topk_sim_matrix,idx_near = torch.topk(sim_matrix,k,dim=1)  #batch x K
    scores_near = scores[idx_near].detach().clone()  #batch x K x num_class
    diff_scores = torch.sum((p.unsqueeze(1) - scores_near)**2,-1)
    
    loss = -1.0* topk_sim_matrix * diff_scores
    return loss.mean()
    
    
def knn_affinity(X,knn):
    #x [N,D]
    N = X.size(0)
    X = F.normalize(X,1)
    dist = torch.norm(X.unsqueeze(0) - X.unsqueeze(1), dim=-1, p=2)  # [N, N]
    n_neighbors = min(knn + 1, N)
    knn_index = dist.topk(n_neighbors, -1, largest=False).indices[:, 1:]  # [N, knn]
    W = torch.zeros(N, N, device=X.device)
    W.scatter_(dim=-1, index=knn_index, value=1.0)
    return W
    
       
def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)

    mse_loss = (input_softmax-target_softmax)**2
    return mse_loss

    
def softmax_kl_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns KL divergence

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_log_softmax = F.log_softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)

    kl_div = F.kl_div(input_log_softmax, target_softmax, reduction='none')
    return kl_div        
        

def get_distances(X, Y, dist_type="cosine"):
    """
    Args:
        X: (N, D) tensor
        Y: (M, D) tensor
    """
    if dist_type == "euclidean":
        distances = torch.cdist(X, Y)
    elif dist_type == "cosine":
        distances = 1 - torch.matmul(F.normalize(X, dim=1), F.normalize(Y, dim=1).T)
    else:
        raise NotImplementedError(f"{dist_type} distance not implemented.")

    return distances


@torch.no_grad()
def soft_k_nearest_neighbors(features, features_bank, probs_bank):
    pred_probs = []
    K = 4
    for feats in features.split(64):
        distances = get_distances(feats, features_bank,"cosine")
        _, idxs = distances.sort()
        idxs = idxs[:, : K]
        # (64, num_nbrs, num_classes), average over dim=1
        probs = probs_bank[idxs, :].mean(1)
        pred_probs.append(probs)
    pred_probs = torch.cat(pred_probs)
    _, pred_labels = pred_probs.max(dim=1)

    return pred_labels, pred_probs
