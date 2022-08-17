import pickle

def save(self,save_path="latest.pkl"):
    print("即将开始保存模型...")
    f = open(save_path,"wb")
    pickle.dump(self.model,f)
    f.close()
    print("保存模型  ",save_path,"  成功！")

def load_model(self,load_path="latest.pkl"):
    print("即将开始加载模型...")
    f = open(load_path,"rb")
    self.model = pickle.load(f)
    f.close()
    print("加载模型  ",load_path,"  成功！")