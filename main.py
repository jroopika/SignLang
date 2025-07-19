from utils.load_data import get_data_loaders

train_loader, classes = get_data_loaders("dataset/asl_alphabet_train", batch_size=32)

print("Loaded classes:", classes)

# Test printing one batch shape
for images, labels in train_loader:
    print("Batch shape:", images.shape)
    print("Labels:", labels[:10])
    break
