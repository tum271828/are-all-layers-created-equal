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
    
    def __init__(self,purpose,trainDs,testDs,modelname,epochs=100,lr=0.0001,showRand=False):
        def forceDir(path):
            if not os.path.exists(path):
                os.mkdir(path)
        
        self.purpose=purpose
        self.trainDs=trainDs 
        self.testDs=testDs
        self.epochs=epochs
        self.depth=5
        self.period=epochs//5
        self.modelname=modelname
        self.inputShape=self.trainDs[0][0].shape
        self.lr=lr
        self.showRand=showRand
        self.monitor="val_acc"
        print("input shape",self.inputShape)
        
        forceDir("output")
        forceDir("data")

    def on_train_begin(self,*argv,**kargv):
        self.initWeight=self.getWeightAsDict(self.model)        
        self.stats=[]
        
    def on_epoch_end(self,epoch,logs=None):
        #return
        if epoch%self.period!=0: return
        curWeight=self.getWeightAsDict(self.model)
        stats={}
        for name,w in curWeight.items():
            level=name # int(name.replace("layer_",""))
            stat=[logs.get("val_loss"),logs.get(self.monitor)]
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
        if self.modelname=="ae":
            import autoencoder
            x=self.trainDs[0].reshape((-1,28,28,1))
            x2=self.testDs[0].reshape((-1,28,28,1))
            x=x/255
            x2=x2/255
            def acc(y_true,y_pred):
                return 1-K.mean(K.abs(y_true-y_pred))
            self.trainDs=(x,x)
            self.testDs=(x2,x2)
            model,_,_=autoencoder.autoencoder(self.inputShape[-1],neuron=8)
            model.compile(loss="mse",optimizer="adam",metrics=[acc])
            return model
        elif self.modelname=="mobilenetv2":  
            if len(self.inputShape)==2:
                x=Reshape(self.inputShape+(1,))(input)
            else:
                x=input
            basemodel=ka.mobilenet_v2.MobileNetV2(weights=None,include_top=False,input_tensor=x,classes=10) 
            #input=basemodel.input
            #print(basemodel.layers[(-15-6-6-6-6)*2-1].name)
            #exit()
            x=basemodel.get_layer("block_5_add").output
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
                    optimizer=keras.optimizers.SGD(momentum=0.9,lr=0.00001), #keras.optimizers.Adam(),
                    metrics=['acc'])
        return model
                     
    def getWeightAsDict(self,model):
        
        def allNode(model):
            ans=[]
            if isinstance(model,Model):
                for l in model.layers:
                    ans.extend(allNode(l))
            else:
                if isinstance(model,Conv2D) or isinstance(model,Dense) or isinstance(model,DepthwiseConv2D):
                    ans.append(model)
            return ans
                
        return {l.name:l.get_weights() for l in allNode(model)}

    def setWeight(self,name,w):
        
        def allNode(model):
            ans=[]
            if isinstance(model,Model):
                for l in model.layers:
                    ans.extend(allNode(l))
            else:
                if isinstance(model,Conv2D) or isinstance(model,Dense) or isinstance(model,DepthwiseConv2D):
                    ans.append(model)
            return ans
        ll=allNode(self.model)
        [l for l in ll if l.name==name][0].set_weights(w)
        #self.model.get_layer(name).set_weights(w)
        
    def trainModel(self):
        self.model=model=self.createModel()
        args={}
        args["validation_data"]=self.testDs
        res=model.fit(*self.trainDs,batch_size=128,epochs=self.epochs+1,callbacks=[self],**args)
        return model
    
    def eval(self):
        return self.model.evaluate(*self.testDs,verbose=0)        
    
    def plot(self,name,ax):
        dashs=[[1,0],[20,20],[40,40],[80,80]]
        mak=["^","o","."]
        for i,level in enumerate(self.stats[0].keys()):
            #lname=level #int(level.replace("layer_",""))
            #if int(level) % 10!=0: continue
            x,y=[],[]
            period=self.epochs//len(self.stats)
            for ep,stat in enumerate(self.stats): 
                #if ep % period!=0: continue
                loss,acc=stat[level][name]
                _,tacc=stat[level]["trained"]
                x.append(ep*period)
                y.append(acc/tacc)
            ax.plot(x,y,label="re-"+name+" "+level,marker=mak[i%3])
        
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
        
        fig,(ax1)=plt.subplots(nrows=1,ncols=1,figsize=(12,12))
        self.plot("init",ax1)
        if self.showRand:
            self.plot("rand",ax1)
        ax1.set_ylabel("robustness")
        ax1.set_xlabel("ep")        
        ax1.legend(loc=0,fontsize="small")
        #ax2.legend()
        ax1.set_ylim(0,1)        
        #ax1.title("layer robustness [256,256,256,10] on MNIST")
        #ax2.set_ylim(0,1)
        
        fig.savefig("output/{}.png".format(self.purpose))

def experiment(modelname="dense256x3",dsname="mnist",epochs=30,lr=0.0001,show_rand=False):
    """ run experiment
    :param modelname: dense256x3|dense256x5
    :param dsname: mnist|cifar10
    """

    if dsname=="mnist":
        (trainX,trainY),(testX,testY)=keras.datasets.mnist.load_data()
    elif dsname=="cifar10":
        (trainX,trainY),(testX,testY)=keras.datasets.cifar10.load_data()
    else:
        raise ValueError("undefine dsname {}".format(dsname))
    trainY,testY = [keras.utils.to_categorical(y, 10) for y in [trainY,testY]]
    leMea=LayerEqualityMeasurer("{}-{}-{}ep".format(modelname,dsname,epochs),(trainX,trainY),(testX,testY),modelname,epochs=epochs,showRand=show_rand)
    leMea.collect()
    leMea.viz()

fire.Fire(experiment)




