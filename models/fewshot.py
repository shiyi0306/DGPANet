"""
Query-Informed FSS
Extended from ADNet code by Hansen et al.
"""

import torch
import cv2
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from .encoder import Res101Encoder
import matplotlib.pyplot as plt
import numpy as np
import datetime

class FewShotSeg(nn.Module):

    def __init__(self, pretrained_weights="deeplabv3", alpha=0.8):
        super().__init__()

        # Encoder
        self.encoder = Res101Encoder(replace_stride_with_dilation=[True, True, False],
                                     pretrained_weights=pretrained_weights)  # or "resnet101"
        self.device = torch.device('cuda')
        self.scaler = 20.0
        self.criterion = nn.NLLLoss()
        # self.alpha = torch.Tensor([alpha, 1 - alpha])
        self.alpha = torch.Tensor([0.4, 0.5, 0.1])
        # self.alpha = torch.Tensor([1-alpha, alpha])
    def forward(self, supp_imgs, supp_mask, qry_imgs, train=False, n_iters=0):
        """
        Args:
            supp_imgs: support images
                way x shot x [B x 3 x H x W], list of lists of tensors
            fore_mask: foreground masks for support images
                way x shot x [B x H x W], list of lists of tensors
            back_mask: background masks for support images
                way x shot x [B x H x W], list of lists of tensors
            qry_imgs: query images
                N x [B x 3 x H x W], list of tensors
        """

        self.n_ways = len(supp_imgs)
        self.n_shots = len(supp_imgs[0])
        self.n_queries = len(qry_imgs)
        assert self.n_ways == 1  # for now only one-way, because not every shot has multiple sub-images
        assert self.n_queries == 1
        # mkfile_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d%H%M%S')

        qry_bs = qry_imgs[0].shape[0]
        supp_bs = supp_imgs[0][0].shape[0]
        img_size = supp_imgs[0][0].shape[-2:]
        # print('img_size', img_size)
        supp_mask = torch.stack([torch.stack(way, dim=0) for way in supp_mask],
                                dim=0).view(supp_bs, self.n_ways, self.n_shots, *img_size)  # B x Wa x Sh x H x W
        # s_mask = supp_mask.squeeze(1)
        # channel_mean_s_mask = torch.mean(s_mask, dim=1, keepdim=True)
        # channel_mean_s_mask = F.interpolate(s_mask, size=img_size, mode='bilinear', align_corners=False)
        # channel_mean_s_mask = channel_mean_s_mask.squeeze(0).squeeze(0).cpu().numpy()  # 四维压缩为二维
        # channel_mean_s_mask = (((channel_mean_s_mask - np.min(channel_mean_s_mask)) / (
        #         np.max(channel_mean_s_mask) - np.min(channel_mean_s_mask))) * 255).astype(np.uint8)
        # savedir = './figures/'
        # # if not os.path.exists(savedir + 'feature_vis'): os.makedirs(savedir + 'feature_vis')
        # channel_mean_s_mask = cv2.applyColorMap(channel_mean_s_mask, cv2.COLORMAP_JET)
        # cv2.imwrite(savedir+ str('s_mask') + str(mkfile_time) + '.png', channel_mean_s_mask)

        ###### Extract features ######
        imgs_concat = torch.cat([torch.cat(way, dim=0) for way in supp_imgs]
                                + [torch.cat(qry_imgs, dim=0), ], dim=0)

        img_fts, tao = self.encoder(imgs_concat)

        supp_fts = [img_fts[dic][:self.n_ways * self.n_shots * supp_bs].view(  # B x Wa x Sh x C x H' x W'
            supp_bs, self.n_ways, self.n_shots, -1, *img_fts[dic].shape[-2:]) for _, dic in enumerate(img_fts)]
        qry_fts = [img_fts[dic][self.n_ways * self.n_shots * supp_bs:].view(  # B x N x C x H' x W'
            qry_bs, self.n_queries, -1, *img_fts[dic].shape[-2:]) for _, dic in enumerate(img_fts)]
        # for n in range(len(supp_fts)):
            # s_feat = supp_fts[n].squeeze(1).squeeze(1)
            # q_feat = qry_fts[n].squeeze(1)
            # channel_mean_s = torch.mean(s_feat, dim=1, keepdim=True)
            # channel_mean_q = torch.mean(q_feat, dim=1, keepdim=True)
            # channel_mean_s = F.interpolate(channel_mean_s, size=img_size, mode='bilinear', align_corners=False)
            # channel_mean_q = F.interpolate(channel_mean_q, size=img_size, mode='bilinear', align_corners=False)
            # channel_mean_s = channel_mean_s.squeeze(0).squeeze(0).cpu().numpy()  # 四维压缩为二维
            # channel_mean_q = channel_mean_q.squeeze(0).squeeze(0).cpu().numpy()  # 四维压缩为二维
            # channel_mean_s = (((channel_mean_s - np.min(channel_mean_s)) / (
            #         np.max(channel_mean_s) - np.min(channel_mean_s))) * 255).astype(np.uint8)
                  # digits = load_digits()
        # embeddings = TSNE(n_jobs=4).fit_transform(digits.data)
        # vis_x = embeddings[:, 0]
        # vis_y = embeddings[:, 1]
        # plt.scatter(vis_x, vis_y, c=digits.target, cmap=plt.cm.get_cmap("jet", 10), marker='.')
        # plt.colorbar(ticks=range(10))
        # plt.clim(-0.5, 9.5)
        # plt.show()  # channel_mean_q = (((channel_mean_q - np.min(channel_mean_q)) / (
            #         np.max(channel_mean_q) - np.min(channel_mean_q))) * 255).astype(np.uint8)
            # savedir = './figures/'
            # # if not os.path.exists(savedir + 'feature_vis'): os.makedirs(savedir + 'feature_vis')
            # channel_mean_s = cv2.applyColorMap(channel_mean_s, cv2.COLORMAP_JET)
            # channel_mean_q = cv2.applyColorMap(channel_mean_q, cv2.COLORMAP_JET)
            # cv2.imwrite(savedir + str(n) + str('s_feat') + str(mkfile_time) + '.png', channel_mean_s)
            # cv2.imwrite(savedir + str(n) + str('q_feat') + str(mkfile_time) + '.png', channel_mean_q)


        ###### Get threshold #######
        self.t = tao[self.n_ways * self.n_shots * supp_bs:]  # t for query features
        self.thresh_pred = [self.t for _ in range(self.n_ways)]


        ###### Compute loss ######
        align_loss = torch.zeros(1).to(self.device)
        outputs = []
        outputs_ = []
        for epi in range(supp_bs):
            ###### Extract prototypes ######
            supp_fts_ = [[[self.getFeatures(supp_fts[n][[epi], way, shot], supp_mask[[epi], way, shot])
                           for shot in range(self.n_shots)] for way in range(self.n_ways)] for n in
                         range(len(supp_fts))]
            fg_prototypes = [self.getPrototype(supp_fts_[n]) for n in range(len(supp_fts))]
            fp_l = []
            for n in range(len(supp_fts)):
                fp = torch.stack(fg_prototypes[n], dim=0)
                fp_l.append(fp.unsqueeze(-1).unsqueeze(-1))

            # ###### Get query predictions ######
            qry_pred = [torch.stack(
                [self.getPred(qry_fts[n][epi], fg_prototypes[n][way], self.thresh_pred[way])
                 for way in range(self.n_ways)], dim=1) for n in range(len(qry_fts))]  # N x Wa x H' x W'

            # ###### Prototype Refinement (only for test) ######
            fg_prototypes_ = []
            if (not train) and n_iters > 0:  # iteratively update prototypes
                fp_l = []
                for n in range(len(qry_fts)):
                    fg_prototypes_.append(
                        self.updatePrototype(qry_fts[n], fg_prototypes[n], qry_pred[n], n_iters, epi))
                    fp = fg_prototypes_[n]
                    fp_l.append(fp.unsqueeze(-1).unsqueeze(-1))

                qry_pred = [torch.stack(
                    [self.getPred(qry_fts[n][epi], fg_prototypes_[n][way], self.thresh_pred[way]) for way in
                     range(self.n_ways)], dim=1) for n in range(len(qry_fts))]  # N x Wa x H' x W'


            fp_ls = []
            bp_ls = []
            for n in range(len(qry_fts)):
                AFP, ABP, LBP = self.AP(qry_fts[n][epi], qry_pred[n], epi)
                fp = fp_l[n] * 0.9 + AFP * 0.1
                # print('fp.shape', fp.shape)
                bp = ABP * 0.2 + LBP * 0.8
                # print('bp.shape', bp.shape)
                fp_ls.append(fp)
                bp_ls.append(bp)

            #############################################################
            # qry_pred_up_ = [F.interpolate(qry_pred[n], size=img_size, mode='bilinear', align_corners=True)
            #                 for n in range(len(qry_fts))]
            # pred_ = [self.alpha[n] * qry_pred_up_[n] for n in range(len(qry_fts))]
            # preds_ = torch.sum(torch.stack(pred_, dim=0), dim=0) / torch.sum(self.alpha)
            # preds_bg_ = 1 - preds_
            # channel_mean_fg_ = F.interpolate(preds_, size=img_size, mode='bilinear', align_corners=False)
            # channel_mean_bg_ = F.interpolate(preds_bg_, size=img_size, mode='bilinear', align_corners=False)
            # channel_mean_fg_ = channel_mean_fg_.squeeze(0).squeeze(0).cpu().numpy()  # 四维压缩为二维
            # channel_mean_bg_ = channel_mean_bg_.squeeze(0).squeeze(0).cpu().numpy()  # 四维压缩为二维
            # channel_mean_fg_ = (((channel_mean_fg_ - np.min(channel_mean_fg_)) / (
            #         np.max(channel_mean_fg_) - np.min(channel_mean_fg_))) * 255).astype(np.uint8)
            # channel_mean_bg_ = (((channel_mean_bg_ - np.min(channel_mean_bg_)) / (
            #         np.max(channel_mean_bg_) - np.min(channel_mean_bg_))) * 255).astype(np.uint8)
            # savedir = './figures/'
            # # if not os.path.exists(savedir + 'feature_vis'): os.makedirs(savedir + 'feature_vis')
            # channel_mean_fg_ = cv2.applyColorMap(channel_mean_fg_, cv2.COLORMAP_JET)
            # channel_mean_bg_ = cv2.applyColorMap(channel_mean_bg_, cv2.COLORMAP_JET)
            # cv2.imwrite(savedir + str('fg_Q-Net_pred') + str(mkfile_time) + '.png', channel_mean_fg_)
            # cv2.imwrite(savedir + str('bg_Q-Net_pred') + str(mkfile_time) + '.png', channel_mean_bg_)
            ###########################################################


            qry_pred_new_fg = [torch.stack(
                [self.getSim(qry_fts[n][epi], fp_ls[n][way], self.thresh_pred[way]) for way in
                 range(fp_ls[n].shape[0])], dim=1) for n in range(len(qry_fts))]  # N x Wa x H' x W'
            qry_pred_new_bg = [torch.stack(
                [self.getSim(qry_fts[n][epi], bp_ls[n][way], self.thresh_pred[way]) for way in
                 range(bp_ls[n].shape[0])], dim=1) for n in range(len(qry_fts))]  # N x 1 x H' x W'  way=1

            # ####### Combine predictions of different feature maps ######
            qry_pred_up = [F.interpolate(qry_pred_new_fg[n], size=img_size, mode='bilinear', align_corners=True)
                           for n in range(len(qry_fts))]
            qry_pred_bg_up = [F.interpolate(qry_pred_new_bg[n], size=img_size, mode='bilinear', align_corners=True)
                            for n in range(len(qry_fts))]
            pred = [self.alpha[n] * qry_pred_up[n] for n in range(len(qry_fts))]
            pred_bg = [self.alpha[n] * qry_pred_bg_up[n] for n in range(len(qry_fts))]
            preds = torch.sum(torch.stack(pred, dim=0), dim=0) / torch.sum(self.alpha)  # N x Wa x H' x W'
            preds_bg = torch.sum(torch.stack(pred_bg, dim=0), dim=0) / torch.sum(self.alpha)
            # arr_save = preds.cpu().numpy()
            # file = './heatmap/heatmap.npy'
            # np.save(file, arr_save, allow_pickle=True)
            # s_feat = supp_fts[n].squeeze(1).squeeze(1)
            # q_feat = qry_fts[n].squeeze(1)
            # channel_mean_s = torch.mean(s_feat, dim=1, keepdim=True)
            # channel_mean_q = torch.mean(q_feat, dim=1, keepdim=True)
            # channel_mean_fg = F.interpolate(preds, size=img_size, mode='bilinear', align_corners=False)
            # channel_mean_bg = F.interpolate(preds_bg, size=img_size, mode='bilinear', align_corners=False)
            # channel_mean_fg = channel_mean_fg.squeeze(0).squeeze(0).cpu().numpy()  # 四维压缩为二维
            # channel_mean_bg = channel_mean_bg.squeeze(0).squeeze(0).cpu().numpy()  # 四维压缩为二维
            # channel_mean_fg = (((channel_mean_fg - np.min(channel_mean_fg)) / (
            #         np.max(channel_mean_fg) - np.min(channel_mean_fg))) * 255).astype(np.uint8)
            # channel_mean_bg = (((channel_mean_bg - np.min(channel_mean_bg)) / (
            #         np.max(channel_mean_bg) - np.min(channel_mean_bg))) * 255).astype(np.uint8)
            # savedir = './figures/'
            # # if not os.path.exists(savedir + 'feature_vis'): os.makedirs(savedir + 'feature_vis')
            # channel_mean_fg = cv2.applyColorMap(channel_mean_fg, cv2.COLORMAP_JET)
            # channel_mean_bg = cv2.applyColorMap(channel_mean_bg, cv2.COLORMAP_JET)
            # cv2.imwrite(savedir + str('fg_pred') + str(mkfile_time) + '.png', channel_mean_fg)
            # cv2.imwrite(savedir + str('bg_pred') + str(mkfile_time) + '.png', channel_mean_bg)
            preds = torch.cat((preds_bg, preds), dim=1)# N x (1 + Wa) x H x W
            # heat = preds
            preds = preds.softmax(1)
            outputs.append(preds)

            qry_pred_up_ = [F.interpolate(qry_pred[n], size=img_size, mode='bilinear', align_corners=True)
                            for n in range(len(qry_fts))]
            pred_ = [self.alpha[n] * qry_pred_up_[n] for n in range(len(qry_fts))]
            preds_ = torch.sum(torch.stack(pred_, dim=0), dim=0) / torch.sum(self.alpha)
            # preds_bg_ = 1-pred_
            # channel_mean_fg_ = F.interpolate(preds_, size=img_size, mode='bilinear', align_corners=False)
            # channel_mean_bg_ = F.interpolate(preds_bg_, size=img_size, mode='bilinear', align_corners=False)
            # channel_mean_fg_ = channel_mean_fg_.squeeze(0).squeeze(0).cpu().numpy()  # 四维压缩为二维
            # channel_mean_bg_ = channel_mean_bg_.squeeze(0).squeeze(0).cpu().numpy()  # 四维压缩为二维
            # channel_mean_fg_ = (((channel_mean_fg_ - np.min(channel_mean_fg_)) / (
            #         np.max(channel_mean_fg_) - np.min(channel_mean_fg_))) * 255).astype(np.uint8)
            # channel_mean_bg_ = (((channel_mean_bg_ - np.min(channel_mean_bg_)) / (
            #         np.max(channel_mean_bg_) - np.min(channel_mean_bg_))) * 255).astype(np.uint8)
            # savedir = './figures/'
            # if not os.path.exists(savedir + 'feature_vis'): os.makedirs(savedir + 'feature_vis')
            # channel_mean_fg_ = cv2.applyColorMap(channel_mean_fg_, cv2.COLORMAP_JET)
            # channel_mean_bg_ = cv2.applyColorMap(channel_mean_bg_, cv2.COLORMAP_JET)
            # cv2.imwrite(savedir + str('fg_Q-Net_pred') + str(mkfile_time) + '.png', channel_mean_fg_)
            # cv2.imwrite(savedir + str('bg_Q-Net_pred') + str(mkfile_time) + '.png', channel_mean_bg_)
            preds_ = torch.cat((1.0 - preds_, preds_), dim=1)  # N x (1 + Wa) x H x W
            outputs_.append(preds_)

            ###### Prototype alignment loss ######
            if train:
                align_loss_epi = self.alignDualLoss([supp_fts[n][epi] for n in range(len(supp_fts))],
                                                [qry_fts[n][epi] for n in range(len(qry_fts))],
                                                preds, supp_mask[epi], epi)
                # align_loss_epi = self.alignLoss([supp_fts[n][epi] for n in range(len(supp_fts))],
                #                                 [qry_fts[n][epi] for n in range(len(qry_fts))],
                #                                 preds, supp_mask[epi])
                fp_loss_epi = self.fgProtoLoss([fg_prototypes[n] for n in range(len(supp_fts))],
                                              [supp_fts[n][epi] for n in range(len(supp_fts))],
                                              supp_mask[epi])
                proto_loss_epi = self.protoLoss([fp_ls[n] for n in range(len(qry_fts))],
                                              [bp_ls[n] for n in range(len(qry_fts))],
                                              [supp_fts[n][epi] for n in range(len(supp_fts))],
                                              supp_mask[epi])
                # align_loss_epi = align_loss_epi + align_loss_epi_
                align_loss_epi = 0.2 * align_loss_epi + 0.4 * proto_loss_epi + 0.4 * fp_loss_epi
                align_loss += align_loss_epi

        output = torch.stack(outputs, dim=1)  # N x B x (1 + Wa) x H x W
        output = output.view(-1, *output.shape[2:])
        output_ = torch.stack(outputs_, dim=1)  # N x B x (1 + Wa) x H x W
        output_ = output_.view(-1, *output_.shape[2:])
        output = 0.5 * output + 0.5 * output_
        return output, align_loss / supp_bs



        # return output, align_loss / supp_bs
    def updatePrototype(self, fts, prototype, pred, update_iters, epi):

        prototype_0 = torch.stack(prototype, dim=0)
        n_ways = len(prototype)
        prototype_ = Parameter(torch.stack(prototype, dim=0))

        optimizer = torch.optim.Adam([prototype_], lr=0.01)

        while update_iters > 0:
            with torch.enable_grad():
                pred_mask = torch.sum(pred, dim=-3)
                pred_mask = torch.stack((1.0 - pred_mask, pred_mask), dim=1).argmax(dim=1, keepdim=True)
                pred_mask = pred_mask.repeat([*fts.shape[1:-2], 1, 1])
                bg_fts = fts[epi] * (1 - pred_mask)
                fg_fts = torch.zeros_like(fts[epi])
                for way in range(n_ways):
                    fg_fts += prototype_[way].unsqueeze(-1).unsqueeze(-1).repeat(*pred.shape) \
                              * pred_mask[way][None, ...]
                new_fts = bg_fts + fg_fts
                fts_norm = torch.sigmoid((fts[epi] - fts[epi].min()) / (fts[epi].max() - fts[epi].min()))
                new_fts_norm = torch.sigmoid((new_fts - new_fts.min()) / (new_fts.max() - new_fts.min()))
                bce_loss = nn.BCELoss()
                loss = bce_loss(fts_norm, new_fts_norm)


            optimizer.zero_grad()
            # loss.requires_grad_()
            loss.backward()
            optimizer.step()

            pred = torch.stack([self.getPred(fts[epi], prototype_[way], self.thresh_pred[way])
                                for way in range(self.n_ways)], dim=1)  # N x Wa x H' x W'
            update_iters += -1

        return prototype_

    def getPred(self, fts, prototype, thresh):
        """
        Calculate the distance between features and prototypes

        Args:
            fts: input features
                expect shape: N x C x H x W
            prototype: prototype of one semantic class
                expect shape: 1 x C
        """
        # print('thresh', thresh)
        sim = -F.cosine_similarity(fts, prototype[..., None, None], dim=1) * self.scaler
        pred = 1.0 - torch.sigmoid(0.5 * (sim - thresh))
        return pred

    def getSim(self, fts, prototype, thresh):
        """
        Calculate the distance between features and prototypes

        Args:
            fts: input features
                expect shape: N x C x H x W
            prototype: prototype of one semantic class
                expect shape: 1 x C x H x W
        """
        sim = F.cosine_similarity(fts, prototype, dim=1) * self.scaler
        # pred = 1.0 - torch.sigmoid(0.5 * (sim - thresh))
        return sim
    # def sim(self, feature_q, fg_proto, bg_proto):
    #     similarity_fg = F.cosine_similarity(feature_q, fg_proto, dim=1)
    #     similarity_bg = F.cosine_similarity(feature_q, bg_proto, dim=1)
    #     out = torch.cat((similarity_bg[:, None, ...], similarity_fg[:, None, ...]), dim=1) * 4.0
    #     out = out.softmax(1)
    #     return out

    def AP(self, feature_q, out, epi):
        channel = feature_q.shape[1]
        n_queries = feature_q.shape[0]
        n_ways = out.shape[1]
        pred_fg = out.view(n_queries, n_ways, -1)
        pred_bg = (1 - out).view(n_queries, 1, -1)
        fg_ls = []
        bg_ls = []
        # fg_local_ls = []
        bg_local_ls = []
        mkfile_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d%H%M%S')
        for n in range(n_queries):
            cur_feat = feature_q[n].view(channel, -1)
            # print('cur_feat.shape', cur_feat.shape)
            f_h, f_w = feature_q[n].shape[-2:]
            fg_l = []
            bg_l = []
            # fg_local_l = []
            bg_local_l = []

            for way in range(n_ways):
                fg_feat = cur_feat[:, torch.topk(pred_fg[epi][way], 12).indices]
                # if (pred_fg[epi][way] > 0.8).sum() > 0:
                #     fg_feat = cur_feat[:, (pred_fg[epi][way] > 0.8)]  # .mean(-1)
                # # elif (pred_fg[epi][way] > 0.7).sum() > 0:
                # #         fg_feat = cur_feat[:, (pred_fg[epi][way] > 0.7)]  # .mean(-1)
                # elif (pred_fg[epi][way] > 0.6).sum() > 0:
                #         fg_feat = cur_feat[:, (pred_fg[epi][way] > 0.6)]  # .mean(-1)
                # else:
                #     # if (pred_fg[epi][way] > 0.6).sum() > 0:
                #     #     fg_feat = cur_feat[:, (pred_fg[epi][way] > 0.6)]  # .mean(-1)
                #     # else:
                #     #     fg_feat = cur_feat[:, torch.topk(pred_fg[epi][way], 12).indices]  # .mean(-1)
                #     fg_feat = cur_feat[:, torch.topk(pred_fg[epi][way], 12).indices]  # .mean(-1)
                fg_proto = fg_feat.mean(-1)
                # proto = fg_feat.t().cpu()


                # pca = PCA(n_components=2)
                # tsne = TSNE(n_components=2)
                # tsne_prototypes = tsne.fit_transform(proto)
                # pca_prototypes = pca.fit_transform(proto)
                # plt.subplot(121)
                # plt.scatter(pca_prototypes[:, 0], pca_prototypes[:, 1])
                # plt.title('PCA fg_proto Visualization')
                # plt.xlabel('Dimension 1')
                # plt.ylabel('Dimension 2')
                # # plt.savefig("./figure/PCA_fg_proto{}.png".format(mkfile_time))
                # plt.subplot(122)
                # plt.scatter(tsne_prototypes[:, 0], tsne_prototypes[:, 1])
                # plt.title('t-SNE Prototype Features')
                # plt.xlabel('Dimension 1')
                # plt.ylabel('Dimension 2')
                # plt.savefig("./figure/PCA_TSNE_fg_proto{}.png".format(mkfile_time))
                # plt.savefig("fg_proto.png")

                fg_l.append(fg_proto)
            cur_feat_norm = cur_feat / torch.norm(cur_feat, 2, 0, True)  # 1024, N3
            cur_feat_norm_t = cur_feat_norm.t()  # N3, 1024
            # proto = cur_feat_norm_t.cpu()
            # pca = PCA(n_components=2)
            # tsne = TSNE(n_components=2, random_state=42)
            # pca_prototypes = pca.fit_transform(proto)
            # tsne_prototypes = tsne.fit_transform(proto)
            # plt.subplot(121)
            # plt.scatter(pca_prototypes[:, 0], pca_prototypes[:, 1])
            # plt.title('PCA fg_proto Visualization')
            # plt.xlabel('Dimension 1')
            # plt.ylabel('Dimension 2')
            # # plt.savefig("./figure/PCA_fg_proto{}.png".format(mkfile_time))
            # plt.subplot(122)
            # plt.scatter(tsne_prototypes[:, 0], tsne_prototypes[:, 1])
            # plt.title('t-SNE Prototype Features')
            # plt.xlabel('Dimension 1')
            # plt.ylabel('Dimension 2')
            # plt.savefig("./figure/PCA_TSNE_cur_feat{}.png".format(mkfile_time))
            # plt.savefig("./figures/cur_feat{}.png".format(mkfile_time))
            # plt.savefig("cur_feat.png")

            for way in range(pred_bg.shape[1]):
                bg_feat = cur_feat[:, torch.topk(pred_bg[epi][way], 12).indices]
                # if (pred_bg[epi][way] > 0.7).sum() > 0:
                #     bg_feat = cur_feat[:, (pred_bg[epi][way] > 0.7)]  # .mean(-1)
                # # elif (pred_bg[epi][way] > 0.6).sum() > 0:
                # #     bg_feat = cur_feat[:, (pred_bg[epi][way] > 0.6)]  # .mean(-1)
                # elif (pred_bg[epi][way] > 0.5).sum() > 0:
                #     bg_feat = cur_feat[:, (pred_bg[epi][way] > 0.5)]  # .mean(-1)
                # else:
                #     bg_feat = cur_feat[:, torch.topk(pred_bg[epi][way], 12).indices]  # .mean(-1)
                # else:
                #     if (pred_bg[epi][way] > 0.5).sum() > 0:
                #         bg_feat = cur_feat[:, (pred_bg[epi][way] > 0.5)]  # .mean(-1)
                #     else:
                #         bg_feat = cur_feat[:, torch.topk(pred_bg[epi][way], 12).indices]  # .mean(-1)
                #     # bg_feat = cur_feat[:, torch.topk(pred_bg[epi][way], 12).indices]  # .mean(-1)
                bg_proto = bg_feat.mean(-1)
                # proto = bg_feat.t().cpu()
                # pca = PCA(n_components=2)
                # tsne = TSNE(n_components=2, random_state=42)
                # pca_prototypes = pca.fit_transform(proto)
                # tsne_prototypes = tsne.fit_transform(proto)
                # plt.subplot(121)
                # plt.scatter(pca_prototypes[:, 0], pca_prototypes[:, 1])
                # plt.title('PCA fg_proto Visualization')
                # plt.xlabel('Dimension 1')
                # plt.ylabel('Dimension 2')
                # # plt.savefig("./figure/PCA_fg_proto{}.png".format(mkfile_time))
                # plt.subplot(122)
                # plt.scatter(tsne_prototypes[:, 0], tsne_prototypes[:, 1])
                # plt.title('t-SNE Prototype Features')
                # plt.xlabel('Dimension 1')
                # plt.ylabel('Dimension 2')
                # plt.savefig("./figure/PCA_TSNE_bg_proto{}.png".format(mkfile_time))
                # plt.savefig("./figures/bg_proto{}.png".format(mkfile_time))
                # plt.savefig("bg_proto.png")
                bg_l.append(bg_proto)
                bg_feat_norm = bg_feat / torch.norm(bg_feat, 2, 0, True)  # 1024, N2
                bg_sim = torch.matmul(cur_feat_norm_t, bg_feat_norm) * 2.0  # N3, N2
                bg_sim = bg_sim.softmax(-1)
                bg_proto_local = torch.matmul(bg_sim, bg_feat.t())  # N3, 1024
                # print('bg_proto_local', bg_proto_local.shape)
                # proto = bg_proto_local.cpu()
                # pca = PCA(n_components=2)
                # tsne = TSNE(n_components=2, random_state=42)
                # pca_prototypes = pca.fit_transform(proto)
                # tsne_prototypes = tsne.fit_transform(proto)
                # plt.subplot(121)
                # plt.scatter(pca_prototypes[:, 0], pca_prototypes[:, 1])
                # plt.title('PCA fg_proto Visualization')
                # plt.xlabel('Dimension 1')
                # plt.ylabel('Dimension 2')
                # # plt.savefig("./figure/PCA_fg_proto{}.png".format(mkfile_time))
                # plt.subplot(122)
                # plt.scatter(tsne_prototypes[:, 0], tsne_prototypes[:, 1])
                # plt.title('t-SNE Prototype Features')
                # plt.xlabel('Dimension 1')
                # plt.ylabel('Dimension 2')
                # plt.savefig("./figure/PCA_TSNE_bg_proto_local{}.png".format(mkfile_time))
                # # plt.savefig("./figure/bg_proto_local{}.png".format(mkfile_time))
                # # plt.savefig("bg_proto_local.png")
                bg_proto_local = bg_proto_local.t().view(channel, f_h, f_w).unsqueeze(0)  # 1024, N3
                bg_local_l.append(bg_proto_local)

            fg_ls.append(torch.stack(fg_l, dim=0).unsqueeze(1))
            bg_ls.append(torch.stack(bg_l, dim=0).unsqueeze(1))
            # fg_local_ls.append(torch.stack(fg_local_l, dim=0))
            bg_local_ls.append(torch.stack(bg_local_l, dim=0))

        # global proto
        new_fg = torch.cat(fg_ls, 1).unsqueeze(-1).unsqueeze(-1)
        new_bg = torch.cat(bg_ls, 1).unsqueeze(-1).unsqueeze(-1)
        # local proto
        # new_fg_local = torch.cat(fg_local_ls, 1)
        new_bg_local = torch.cat(bg_local_ls, 1)
        return new_fg, new_bg, new_bg_local
    def getFeatures(self, fts, mask):
        """
        Extract foreground and background features via masked average pooling

        Args:
            fts: input features, expect shape: 1 x C x H' x W'  (batch个s)
            mask: binary mask, expect shape: 1 x H x W
        """


        fts = F.interpolate(fts, size=mask.shape[-2:], mode='bilinear')

        # masked fg features
        masked_fts = torch.sum(fts * mask[None, ...], dim=(-2, -1)) \
                     / (mask[None, ...].sum(dim=(-2, -1)) + 1e-5)  # 1 x C

        return masked_fts

    def getPrototype(self, fg_fts):
        """
        Average the features to obtain the prototype

        Args:
            fg_fts: lists of list of foreground features for each way/shot
                expect shape: Wa x Sh x [1 x C]
        """

        n_ways, n_shots = len(fg_fts), len(fg_fts[0])
        fg_prototypes = [torch.sum(torch.cat([tr for tr in way], dim=0), dim=0, keepdim=True) / n_shots for way in
                         fg_fts]  ## concat all fg_fts
        return fg_prototypes


    def alignDualLoss(self, supp_fts, qry_fts, pred, fore_mask, epi):
        n_ways, n_shots = len(fore_mask), len(fore_mask[0])

        # Get query mask
        pred_mask = pred.argmax(dim=1, keepdim=True).squeeze(1)  # N x H' x W'
        binary_masks = [pred_mask == i for i in range(1 + n_ways)]
        skip_ways = [i for i in range(n_ways) if binary_masks[i + 1].sum() == 0]
        pred_mask = torch.stack(binary_masks, dim=0).float()  # (1 + Wa) x N x H' x W'

        # Compute the support loss
        loss = torch.zeros(1).to(self.device)
        for way in range(n_ways):
            if way in skip_ways:
                continue
            # Get the query prototypes
            for shot in range(n_shots):
                # Get prototypes
                qry_fts_ = [[self.getFeatures(qry_fts[n], pred_mask[way + 1])] for n in range(len(qry_fts))]
                fg_prototypes = [self.getPrototype([qry_fts_[n]]) for n in range(len(supp_fts))]
                # print('fg_prototypes.shape', fg_prototypes[0].shape)

                # Get predictions
                supp_pred = [self.getPred(supp_fts[n][way, [shot]], fg_prototypes[n][way], self.thresh_pred[way])
                             for n in range(len(supp_fts))]  # N x Wa x H' x W'
                # print('supp_pred.shape', supp_pred[0].shape)
                fp_l = []
                for n in range(len(supp_fts)):
                    fp = torch.stack(fg_prototypes[n], dim=0)
                    # print('fp.shape', fp.shape)
                    fp_l.append(fp.unsqueeze(-1).unsqueeze(-1))
                fp_ls = []
                bp_ls = []
                for n in range(len(qry_fts)):
                    AFP, ABP, LBP = self.AP(supp_fts[n][way, [shot]], supp_pred[n].unsqueeze(0), epi)
                    fp = fp_l[n] * 0.9 + AFP * 0.1
                    bp = ABP * 0.2 + LBP * 0.8
                    fp_ls.append(fp)
                    bp_ls.append(bp)

                supp_pred_new_fg = [torch.stack(
                    [self.getSim(supp_fts[n][way, [shot]], fp_ls[n][way], self.thresh_pred[way]) for way in
                     range(fp_ls[n].shape[0])], dim=1) for n in range(len(qry_fts))]  # N x Wa x H' x W'
                supp_pred_new_bg = [torch.stack(
                    [self.getSim(supp_fts[n][way, [shot]], bp_ls[n][way], self.thresh_pred[way]) for way in
                     range(bp_ls[n].shape[0])], dim=1) for n in range(len(qry_fts))]  # N x 1 x H' x W'  way=1

                # ####### Combine predictions of different feature maps ######
                supp_pred_up = [F.interpolate(supp_pred_new_fg[n], size=fore_mask.shape[-2:], mode='bilinear', align_corners=True)
                               for n in range(len(qry_fts))]
                # qry_pred_up = [F.interpolate(qry_pred[n], size=img_size, mode='bilinear', align_corners=True)
                #               for n in range(len(qry_fts))]
                supp_pred_bg_up = [F.interpolate(supp_pred_new_bg[n], size=fore_mask.shape[-2:], mode='bilinear', align_corners=True)
                                  for n in range(len(qry_fts))]
                pred = [self.alpha[n] * supp_pred_up[n] for n in range(len(qry_fts))]
                pred_bg = [self.alpha[n] * supp_pred_bg_up[n] for n in range(len(qry_fts))]
                preds = torch.sum(torch.stack(pred, dim=0), dim=0) / torch.sum(self.alpha)  # N x Wa x H' x W'
                preds_bg = torch.sum(torch.stack(pred_bg, dim=0), dim=0) / torch.sum(self.alpha)
                preds = torch.cat((preds_bg, preds), dim=1)  # N x (1 + Wa) x H x W
                pred_ups = preds.softmax(1)

                # Construct the support Ground-Truth segmentation
                supp_label = torch.full_like(fore_mask[way, shot], 255, device=fore_mask.device)
                supp_label[fore_mask[way, shot] == 1] = 1
                supp_label[fore_mask[way, shot] == 0] = 0

                # Compute Loss
                eps = torch.finfo(torch.float32).eps
                log_prob = torch.log(torch.clamp(pred_ups, eps, 1 - eps))
                loss += self.criterion(log_prob, supp_label[None, ...].long()) / n_shots / n_ways

        return loss

    def fgProtoLoss(self, fg_prototypes, supp_fts, fore_mask):
        n_ways, n_shots = len(fore_mask), len(fore_mask[0])

        # Compute the support loss
        loss = torch.zeros(1).to(self.device)
        for way in range(n_ways):
            # Get the query prototypes
            for shot in range(n_shots):
                # Get prototypes
                # Get predictions
                supp_pred = [self.getPred(supp_fts[n][way, [shot]], fg_prototypes[n][way], self.thresh_pred[way])
                             for n in range(len(supp_fts))]  # N x Wa x H' x W'
                # supp_pred_bg = [self.getPred_(supp_fts[n][way, [shot]], bg_prototypes[n][way], self.thresh_pred[way])
                #              for n in range(len(supp_fts))]  # N x Wa x H' x W'
                supp_pred = [F.interpolate(supp_pred[n][None, ...], size=fore_mask.shape[-2:], mode='bilinear',
                                           align_corners=True)
                             for n in range(len(supp_fts))]

                # Combine predictions of different feature maps
                preds = [self.alpha[n] * supp_pred[n] for n in range(len(supp_fts))]
                preds = torch.sum(torch.stack(preds, dim=0), dim=0) / torch.sum(self.alpha)
                pred_ups = torch.cat((1-preds, preds), dim=1)

                # Construct the support Ground-Truth segmentation
                supp_label = torch.full_like(fore_mask[way, shot], 255, device=fore_mask.device)
                supp_label[fore_mask[way, shot] == 1] = 1
                supp_label[fore_mask[way, shot] == 0] = 0

                # Compute Loss
                eps = torch.finfo(torch.float32).eps
                log_prob = torch.log(torch.clamp(pred_ups, eps, 1 - eps))
                loss += self.criterion(log_prob, supp_label[None, ...].long()) / n_shots / n_ways

        return loss
    def protoLoss(self, fg, bg, supp_fts, fore_mask):
        n_ways, n_shots = len(fore_mask), len(fore_mask[0])

        # Compute the support loss
        loss = torch.zeros(1).to(self.device)
        for way in range(n_ways):
            # Get the query prototypes
            for shot in range(n_shots):
                # Get prototypes
                # Get predictions
                supp_pred_fg = [self.getSim(supp_fts[n][way, [shot]], fg[n][way], self.thresh_pred[way])
                             for n in range(len(supp_fts))]  # N x Wa x H' x W'
                supp_pred_bg = [self.getSim(supp_fts[n][way, [shot]], bg[n][way], self.thresh_pred[way])
                             for n in range(len(supp_fts))]  # N x Wa x H' x W'
                supp_pred_fg = [F.interpolate(supp_pred_fg[n][None, ...], size=fore_mask.shape[-2:], mode='bilinear',
                                           align_corners=True)
                             for n in range(len(supp_fts))]
                supp_pred_bg = [F.interpolate(supp_pred_bg[n][None, ...], size=fore_mask.shape[-2:], mode='bilinear',
                                           align_corners=True)
                             for n in range(len(supp_fts))]

                # Combine predictions of different feature maps
                preds_fg = [self.alpha[n] * supp_pred_fg[n] for n in range(len(supp_fts))]
                preds_fg = torch.sum(torch.stack(preds_fg, dim=0), dim=0) / torch.sum(self.alpha)
                preds_bg = [self.alpha[n] * supp_pred_bg[n] for n in range(len(supp_fts))]
                preds_bg = torch.sum(torch.stack(preds_bg, dim=0), dim=0) / torch.sum(self.alpha)
                pred_ups = torch.cat((preds_bg, preds_fg), dim=1)
                pred_ups = pred_ups.softmax(1)

                # Construct the support Ground-Truth segmentation
                supp_label = torch.full_like(fore_mask[way, shot], 255, device=fore_mask.device)
                supp_label[fore_mask[way, shot] == 1] = 1
                supp_label[fore_mask[way, shot] == 0] = 0

                # Compute Loss
                eps = torch.finfo(torch.float32).eps
                log_prob = torch.log(torch.clamp(pred_ups, eps, 1 - eps))
                loss += self.criterion(log_prob, supp_label[None, ...].long()) / n_shots / n_ways

        return loss


    def alignLoss(self, supp_fts, qry_fts, pred, fore_mask):
        n_ways, n_shots = len(fore_mask), len(fore_mask[0])

        # Get query mask
        pred_mask = pred.argmax(dim=1, keepdim=True).squeeze(1)  # N x H' x W'
        binary_masks = [pred_mask == i for i in range(1 + n_ways)]
        skip_ways = [i for i in range(n_ways) if binary_masks[i + 1].sum() == 0]
        pred_mask = torch.stack(binary_masks, dim=0).float()  # (1 + Wa) x N x H' x W'

        # Compute the support loss
        loss = torch.zeros(1).to(self.device)
        for way in range(n_ways):
            if way in skip_ways:
                continue
            # Get the query prototypes
            for shot in range(n_shots):
                # Get prototypes
                qry_fts_ = [[self.getFeatures(qry_fts[n], pred_mask[way + 1])] for n in range(len(qry_fts))]
                fg_prototypes = [self.getPrototype([qry_fts_[n]]) for n in range(len(supp_fts))]

                # Get predictions
                supp_pred = [self.getPred(supp_fts[n][way, [shot]], fg_prototypes[n][way], self.thresh_pred[way])
                             for n in range(len(supp_fts))]  # N x Wa x H' x W'
                supp_pred = [F.interpolate(supp_pred[n][None, ...], size=fore_mask.shape[-2:], mode='bilinear',
                                           align_corners=True)
                             for n in range(len(supp_fts))]

                # Combine predictions of different feature maps
                preds = [self.alpha[n] * supp_pred[n] for n in range(len(supp_fts))]
                preds = torch.sum(torch.stack(preds, dim=0), dim=0) / torch.sum(self.alpha)
                pred_ups = torch.cat((1.0 - preds, preds), dim=1)

                # Construct the support Ground-Truth segmentation
                supp_label = torch.full_like(fore_mask[way, shot], 255, device=fore_mask.device)
                supp_label[fore_mask[way, shot] == 1] = 1
                supp_label[fore_mask[way, shot] == 0] = 0

                # Compute Loss
                eps = torch.finfo(torch.float32).eps
                log_prob = torch.log(torch.clamp(pred_ups, eps, 1 - eps))
                loss += self.criterion(log_prob, supp_label[None, ...].long()) / n_shots / n_ways

        return loss

    # def feature_vis(feats):  # feaats形状: [b,c,h,w]
    #     output_shape = (256, 256)  # 输出形状
    #     channel_mean = torch.mean(feats, dim=1, keepdim=True)  # channel_max,_ = torch.max(feats,dim=1,keepdim=True)
    #     channel_mean = F.interpolate(channel_mean, size=output_shape, mode='bilinear', align_corners=False)
    #     channel_mean = channel_mean.squeeze(0).squeeze(0).cpu().numpy()  # 四维压缩为二维
    #     channel_mean = (((channel_mean - np.min(channel_mean)) / (
    #                 np.max(channel_mean) - np.min(channel_mean))) * 255).astype(np.uint8)
    #     savedir = './figures/'
    #     # if not os.path.exists(savedir + 'feature_vis'): os.makedirs(savedir + 'feature_vis')
    #     channel_mean = cv2.applyColorMap(channel_mean, cv2.COLORMAP_JET)
    #     cv2.imwrite(savedir + '0.png', channel_mean)

