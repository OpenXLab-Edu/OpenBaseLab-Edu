from mmImage import *
image_path = "6.png"
# model = MMImage(method = 'contour')
# model = MMImage(method = 'canny')
# model = MMImage(method = 'blur') #para = ['mean',3]
model = MMImage(method = 'corner')  # para = 0.01)
model.load_image(image_path)
image_out = model.process(save_path = './save.png',para = 0.1)
# 
