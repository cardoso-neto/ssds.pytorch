import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Function
from torch.autograd import Variable
from lib.utils.box_utils import decode,nms
# from lib.utils.nms.nms_wrapper import nms
from lib.utils.timer import Timer

class Detect(Function):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """
    def __init__(self, cfg, priors):
        self.num_classes = cfg.NUM_CLASSES
        self.background_label = cfg.BACKGROUND_LABEL
        self.conf_thresh = cfg.SCORE_THRESHOLD
        self.nms_thresh = cfg.IOU_THRESHOLD 
        self.top_k = cfg.MAX_DETECTIONS 
        self.variance = cfg.VARIANCE
        self.priors = Variable(priors)

    def forward(self, predictions):
        """
        Args:
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors*4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch*num_priors,num_classes]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [1,num_priors,4]
        """
        loc, conf = predictions

        loc_data = loc.data
        conf_data = conf.data
        prior_data = self.priors.data

        num = loc_data.size(0)  # batch size
        num_priors = prior_data.size(0)
        #self.output.zero_()
        if num == 1:
            # size batch x num_classes x num_priors
            conf_preds = conf_data.t().contiguous().unsqueeze(0)
        else:
            conf_preds = conf_data.view(
                num,
                num_priors,
                self.num_classes,
            ).transpose(2, 1)
            #self.output.expand_(num, self.num_classes, self.top_k, 5)
        output = torch.zeros(num, self.num_classes, self.top_k, 5)

        _t = {
            'decode': Timer(),
            'misc': Timer(),
            'box_mask': Timer(),
            'score_mask': Timer(), 
            'nms': Timer(),
            'cpu': Timer(),
            'sort':Timer(),
        }
        gpunms_time = 0
        scores_time=0
        box_time=0
        cpu_tims=0
        sort_time=0
        decode_time=0
        _t['misc'].tic()
        # Decode predictions into bboxes.
        for i in range(num):
            _t['decode'].tic()
            decoded_boxes = decode(loc_data[i], prior_data, self.variance)
            decode_time += _t['decode'].toc()
            # For each class, perform nms
            conf_scores = conf_preds[i].clone()
            for cl in range(1, self.num_classes):
                _t['cpu'].tic()
                c_mask = conf_scores[cl].gt(self.conf_thresh).nonzero().view(-1)
                cpu_tims+=_t['cpu'].toc()
                if c_mask.dim() == 0:
                    continue
                _t['score_mask'].tic()
                scores = conf_scores[cl][c_mask]
                scores_time+=_t['score_mask'].toc()
                if scores.dim() == 0:
                    continue
                _t['box_mask'].tic()
                # l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                # boxes = decoded_boxes[l_mask].view(-1, 4)
                boxes = decoded_boxes[c_mask, :]
                box_time+=_t['box_mask'].toc()
                # idx of highest scoring and non-overlapping boxes per class
                _t['nms'].tic()
                # cls_dets = torch.cat((boxes, scores), 1)
                # _, order = torch.sort(scores, 0, True)
                # cls_dets = cls_dets[order]
                # keep = nms(cls_dets, self.nms_thresh)
                # cls_dets = cls_dets[keep.view(-1).long()]
                
                result = nms(boxes, scores, self.nms_thresh, self.top_k)
                # print('len(result):', len(result))
                # if len(result) == 0:
                #     break
                ids, count = result

                gpunms_time += _t['nms'].toc()
                output[i, cl, :count] = torch.cat((
                    scores[ids[:count]].unsqueeze(1),
                    boxes[ids[:count]],
                ), 1)
        nms_time= _t['misc'].toc()
        # print(nms_time, cpu_tims, scores_time,box_time,gpunms_time)
        # flt = self.output.view(-1, 5)
        # _, idx = flt[:, 0].sort(0)
        # _, rank = idx.sort(0)
        # flt[(rank >= self.top_k).unsqueeze(1).expand_as(flt)].fill_(0)
        return output