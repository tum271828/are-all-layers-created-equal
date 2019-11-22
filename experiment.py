import keras.backend as K
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob,cv2,json,tqdm,fire,random,math
import keras
from keras.layers import *
from keras.models import * 
from keras.callbacks import * 
import keras.applications as ka

def randLike(w):
    if isinstance(w,list):
        return [randLike(w_) for w_ in w]
    else:
        return np.random.randn(*w.shape)
        
def calDistance(x,y):
    if isinstance(x,list):
        return sum([calDistance(x_,y_) for x_,y_ in zip(x,y)])
    else:
        return np.linalg.norm(np.squeeze(x-y))
    
class LayerEqualityMeasurer(Callback):
    
    def __init__(self,purpose,trainDs,testDs,modelname,epochs=100,lr=0.0001):
        def forceDir(path):
            if not os.path.exists(path):
                os.mkdir(path)
        
        self.purpose=purpose
        self.trainDs=trainDs 
        self.testDs=testDs
        self.epochs=epochs
        self.modelname=modelname
        self.inputShape=self.trainDs[0][0].shape
        self.lr=lr
        print("input shape",self.inputShape)
        
        forceDir("output")
        forceDir("data")

    def on_train_begin(self,*argv,**kargv):
        self.initWeight=self.getWeightAsDict(self.model)
        self.stats=[]
        
    def on_epoch_end(self,epoch,logs=None):
        #return
        curWeight=self.getWeightAsDict(self.model)
        stats={}
        for name,w in curWeight.items():
            level=name # int(name.replace("layer_",""))
            stat=[logs.get("val_loss"),logs.get("val_acc")]
            initW=self.initWeight[name]
            ans={"trained":stat,"dist":calDistance(w,initW)}
            self.setWeight(name,self.initWeight[name])
            ans["init"]=self.eval()
            self.setWeight(name,randLike(w))
            ans["rand"]=self.eval()
            self.setWeight(name,w) 
            stats[level]=ans
        self.stats.append(stats)
        self.save()
    
    def createModel(self):
        input=Input(shape=self.inputShape)
        if self.modelname=="mobilenetv2":  
            if len(self.inputShape)==2:
                x=Reshape(self.inputShape+(1,))(input)
            basemodel=ka.mobilenet_v2.MobileNetV2(weights=None,include_top=False,input_tensor=x,classes=10) 
            #input=basemodel.input
            x=basemodel.layers[-3].output
            #xGlobalAveragePooling2D
            '''x=Reshape((28,28,1))(input)
            x=Conv2D(32,(3,3),padding="same", activation='relu')(x)
            x=MaxPooling2D()(x)
            x=Conv2D(64,(3,3),padding="same", activation='relu')(x)'''
            x=GlobalAveragePooling2D(name="vv")(x)
        else:
            totalSize=np.prod([s for s in self.inputShape])
            print(totalSize)
            if self.modelname=="dense256x3":
                depth=3
                size=256
            elif self.modelname=="dense256x5":
                depth=5
                size=256
            else:
                raise ValueError("undefined {}".format(self.modelname))
            x=Reshape((totalSize,))(input)
            for d in range(depth):
                x=Dense(size,name="layer_{}".format(d),activation="relu",kernel_initializer="glorot_uniform")(x)
                
        x=Dense(10, activation='softmax',kernel_initializer="glorot_uniform")(x)
        model=Model(input,x)
        
        model.compile(loss='categorical_crossentropy',
                    optimizer=keras.optimizers.SGD(momentum=0.9,lr=self.lr), #keras.optimizers.Adam(),
                    metrics=['acc'])
        return model
                     
    def getWeightAsDict(self,model):
        ans=[]
        def allNode(model):
            if isinstance(model,Model):
                for l in model.layers:
                    ans.extend(allNode(l))
            else:
                return [model]
            return ans
                
        return {l.name:l.get_weights() for l in allNode(model) if "input" not in l.name and "reshape" not in l.name}

    def setWeight(self,name,w):
        self.model.get_layer(name).set_weights(w)
        
    def trainModel(self):
        self.model=model=self.createModel()
        args={}
        args["validation_data"]=self.testDs
        res=model.fit(*self.trainDs,batch_size=128,epochs=self.epochs,callbacks=[self],**args)
        return model
    
    def eval(self):
        return self.model.evaluate(*self.testDs,verbose=0)        
    
    def plot(self,name,ax):
        for level in self.stats[0].keys():
            x,y=[],[]
            period=len(self.stats)//10
            for ep,stat in enumerate(self.stats): 
                if ep % period!=0: continue
                _,acc=stat[level][name]
                _,tacc=stat[level]["trained"]
                x.append(ep)
                y.append(acc/tacc)
            ax.plot(x,y,label="re-"+name+" "+level)
        
    def collect(self):
        filename="data/{}.json".format(self.purpose)
        if not os.path.exists(filename):
            self.trainModel()
    def save(self):
        with open("data/{}.json".format(self.purpose),"w") as fp:
            json.dump(self.stats,fp)
            
    def viz(self):
        with open("data/{}.json".format(self.purpose),"r") as fp:
            self.stats=json.load(fp)
        
        fig,(ax1)=plt.subplots(nrows=1,ncols=1)
        self.plot("init",ax1)
        self.plot("rand",ax1)
        ax1.set_ylabel("robustness")
        ax1.set_xlabel("ep")        
        ax1.legend(loc="center right",fontsize="small")
        ax1.set_ylim(0,1)        
        fig.savefig("output/{}.png".format(self.purpose))

def experiment(modelname="dense256x3",dsname="mnist",epochs=30,lr=0.0001):
    """ run experiment
    :param modelname: dense256x3|dense256x5|mobilenetv2
    :param dsname: mnist
    """

    if dsname=="mnist":
        (trainX,trainY),(testX,testY)=keras.datasets.mnist.load_data()
    else:
        raise ValueError("undefine dsname {}".format(dsname))
    trainY,testY = [keras.utils.to_categorical(y, 10) for y in [trainY,testY]]
    leMea=LayerEqualityMeasurer("{}-{}-{}ep".format(modelname,dsname,epochs),(trainX,trainY),(testX,testY),modelname,epochs=epochs,lr=lr)
    leMea.collect()
    leMea.viz()

fire.Fire(experiment)




