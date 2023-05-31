import numpy as np
import torch


def hit(ng_item, pred_items):
	if ng_item in pred_items:
		return 1
	return 0


def ndcg(ng_item, pred_items):
	if ng_item in pred_items:
		index = pred_items.index(ng_item)
		return np.reciprocal(np.log2(index+2))
	return 0

# def metrics(model, test_loader, top_k, device):
# 	HR, NDCG = [], []
#
# 	for mashup_desc, mashup_desc_len, api_desc,api_desc_len,label,api_index in test_loader:
# 		api_index = api_index.to(device)
#
# 		predictions = model(mashup_desc.to(device), mashup_desc_len.to(device), api_desc.to(device),api_desc_len.to(device))
# 		_, indices = torch.topk(predictions, top_k)
# 		recommends = torch.take(
# 				api_index, indices).cpu().numpy().tolist()
#
# 		ng_item = api_index[0].item() # leave one-out evaluation has only one item per user
# 		HR.append(hit(ng_item, recommends))
# 		NDCG.append(ndcg(ng_item, recommends))
#
# 	return np.mean(HR), np.mean(NDCG)

def metrics(model, test_loader, top_k, device):
	HR, NDCG = [], []

	for mashup_desc, mashup_desc_len, api_desc,api_desc_len,label,api_index in test_loader:
		api_index = api_index.to(device)

		predictions = model(mashup_desc.to(device), mashup_desc_len.to(device), api_desc.to(device),api_desc_len.to(device))
		_, indices = torch.topk(predictions, top_k)
		recommends = torch.take(
				api_index, indices).cpu().numpy().tolist()

		ng_item = api_index[0].item() # leave one-out evaluation has only one item per user
		HR.append(hit(ng_item, recommends))
		NDCG.append(ndcg(ng_item, recommends))

	return np.mean(HR), np.mean(NDCG)