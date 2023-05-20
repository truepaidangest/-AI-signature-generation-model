# “AI签名”生成模型（TensorFlow）                                                    

•	构建GAN网络训练签名图像，冻结discriminator训练generator，从而每次生成不同的签名图像；

•	采用CNN构建generator和discriminator，前者将一个向量转换为候选图像，后者对候选图像进行判别。
