import torch
import torch.nn as nn

class SeasonEmbedding(nn.Module):
    '''
    Embed catagorical variables.
    '''
    def __init__(self):
        super(SeasonEmbedding, self).__init__()
        self.embed_hour=nn.Embedding(26,5)
        #self.embed_week=nn.Embedding(8,3)
        self.embed_week=nn.Embedding(11,3)
        self.embed_month=nn.Embedding(18,3)
        # self.embed_month=nn.Embedding(13,3)
        self.embed_dom=nn.Embedding(32,3) # a typo here but doesn't affect the results. this layer is actually for embedding 24 hours.
        self.embed_season=nn.Embedding(7,5) # a typo here but doesn't affect the results. this layer is actually for embedding 7 days.

    def forward(self, x):
        x_hour = self.embed_hour(x[...,0])
        x_week = self.embed_week(x[...,1])
        x_month = self.embed_month(x[...,2])
        x_dom = self.embed_dom(x[...,3])
        x_season = self.embed_season(x[...,4])
        out=torch.cat((x_hour,x_week,x_month,x_dom,x_season),-1)
        return out
