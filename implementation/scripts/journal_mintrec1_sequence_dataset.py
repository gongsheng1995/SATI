import numpy as np
import torch
def official_mult_dataset_class(parent):
    import numpy as np
    import torch
    class Dataset(parent):
        def __init__(self,*args,**kwargs):
            kwargs['max_length']=50;super().__init__(*args,**kwargs)
            if self.num_classes!=20:raise ValueError('Prepared MulT protocol requires fixed20 MIntRec1 classes.')
            if self.split=='train' and any(row['id'] not in self.prior_dict for row in self.samples):
                raise ValueError('Missing frozen training prior; implicit fallback refused.')
        @staticmethod
        def sequence(feature,cap,dim):
            if isinstance(feature,list):feature=np.concatenate(feature,axis=0)
            value=torch.as_tensor(feature,dtype=torch.float32)
            if value.ndim==3 and value.shape[1]==1:value=value.squeeze(1)
            if value.ndim!=2 or value.shape[-1]!=dim or not len(value) or not torch.isfinite(value).all():
                raise ValueError('Expected finite nonempty original frame sequence with frozen dimensions.')
            result=value.new_full((cap,dim),float('nan'))
            count=min(len(value),cap);result[:count]=value[:count]
            return result
        def _get_audio_feat(self,sid):return self.sequence(self.audio_feats[sid],400,768)
        def _get_video_feat(self,sid):return self.sequence(self.video_feats[sid],180,256)
    return Dataset

