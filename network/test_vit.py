from timm import create_model

# 加载预训练的 ViT-L/14 模型
model = create_model('vit_large_patch14_224', pretrained=False)
print(model)
model.eval()  # 设置为评估模式