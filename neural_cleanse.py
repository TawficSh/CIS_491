import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import random
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from matplotlib import style
style.use('fivethirtyeight')
from tensorflow import keras
from PIL import Image
import os
from sklearn.preprocessing import LabelEncoder


class NeuralCleanse():
    def __init__(self, X, Y, model, num_samples):
        self.X = X
        self.Y = Y
        self.num_classes = int(np.max(Y)+1)
        self.model = model
        self.triggers = []
        self.X_min = np.min(self.X)
        self.X_max = np.max(self.X)
        self.num_samples_per_label = num_samples

    def trigger_insert(self,X,Delta,M):
        return K.clip((1.0-M)*X+M*Delta,self.X_min,self.X_max)

    def draw_trigger(self,M, Delta,file_name):

        plt.cla()
        plt.figure()

        ax = plt.subplot(1,3,1)
        ax.imshow(Delta)
        ax.set_title('Delta')
        ax.set_xticks([])
        ax.set_yticks([])

        ax = plt.subplot(1,3,2)
        ax.imshow(K.reshape(M,(M.shape[0],M.shape[1])),cmap='gray')
        ax.set_title('M')
        ax.set_xticks([])
        ax.set_yticks([])

        ax = plt.subplot(1,3,3)
        ax.imshow(M*Delta)
        ax.set_title('M*Delta')
        ax.set_xticks([])
        ax.set_yticks([])

        plt.tight_layout()
        plt.savefig(file_name+".png")
        plt.close()

    def plot_metrics(self,loss_dict, file_name):
        plt.cla()
        plt.figure()
        ax = plt.subplot(1,1,1)
        for i, metric in enumerate(loss_dict.keys()):
            y = loss_dict[metric]
            x = range(len(y))
            ax.plot(x,y, label=metric)
        ax.set_xlabel('epoch')
        ax.legend(loss_dict.keys(), loc='upper left')
        plt.tight_layout()
        plt.savefig(file_name+".png")
        plt.close()

    def reverse_engineer_triggers(self):

        for target_label in range(self.num_classes):
            print(" ----------------------    reverse engineering the possible trigger for label ", target_label, "   -----------------------")
            ##### Creating X #######
            x_samples = []
            for base_label in range(self.num_classes):
                if base_label == target_label:
                    continue
                possible_idx = (np.where(self.Y==base_label)[0]).tolist()
                idx = random.sample(possible_idx,min(self.num_samples_per_label,len(possible_idx)))
                x_samples.append(self.X[idx,::])

            x_samples = np.vstack(x_samples)
            #y_t = (np.ones((x_samples.shape[0]))*target_label)
            #y_t = keras.utils.to_categorical(y_t,self.num_classes)
            y_t = np.full((x_samples.shape[0], 1), target_label)

            opt_round = 500
            m_opt = keras.optimizers.Adam(learning_rate=0.5)
            delta_opt = keras.optimizers.Adam(learning_rate=0.5)

            m = tf.Variable(np.random.uniform(0.0,1.0,(self.X.shape[1],self.X.shape[2],1)),dtype=tf.float32)
            delta = tf.Variable(np.random.uniform(0.0,1.0,(self.X.shape[1],self.X.shape[2],self.X.shape[3])),dtype=tf.float32)

            no_improvement = 0
            patience = 10
            best_loss = 1e+10
            loss_dict = {'loss':[]}
            lmbda = 0.03
            for r in range(opt_round):

                with tf.GradientTape(persistent=True) as tape:
                 poisoned_x = self.trigger_insert(X=x_samples, Delta=K.sigmoid(delta), M=K.sigmoid(m))
                 prediction = self.model(poisoned_x)
                 individual_losses = []
                 for i in range(delta.shape[-1]):  # Iterate over channels of delta
                  individual_loss = keras.losses.BinaryCrossentropy(from_logits=False)(y_t, prediction)
                  individual_losses.append(individual_loss)
                 loss = tf.reduce_sum(individual_losses) + lmbda * K.sum(K.abs(K.sigmoid(m)))

                #loss = keras.losses.BinaryCrossentropy(from_logits=False)(y_t, prediction) + lmbda * K.sum(K.abs(K.sigmoid(m)))
                #loss = keras.losses.CategoricalCrossentropy()(y_t,prediction) + lmbda*K.sum(K.abs(K.sigmoid(m)))

                if  loss < best_loss:
                    no_improvement = 0
                    best_loss = loss
                else:
                    no_improvement += 1

                if no_improvement == patience:
                    print("\nDecreasing learning rates...")
                    delta_opt.learning_rate = delta_opt.learning_rate/10.0
                    m_opt.learning_rate = m_opt.learning_rate/10.0

                loss_dict['loss'].append(loss)
                print("[",str(r),"] loss:","{0:.5f}".format(loss), end='\r')

                m_grads = tape.gradient(loss,m)
                delta_grads = tape.gradient(loss,delta)

                delta_opt.apply_gradients(zip([delta_grads], [delta]))
                m_opt.apply_gradients(zip([m_grads], [m]))

                if r%50 == 0:
                    if not os.path.isdir("./triggers"):
                        os.system("mkdir ./triggers")

                    self.plot_metrics(loss_dict=loss_dict,file_name='opt-history')
                    self.draw_trigger(M=K.sigmoid(m),Delta=K.sigmoid(delta),file_name=r'C:\Users\ACER\IdeaProjects\ece5831\.triggerstrigger-'+str(target_label))
                    bckdr_acc = self.model.evaluate(self.trigger_insert(X=x_samples,Delta=K.sigmoid(delta), M=K.sigmoid(m)),y_t,verbose=0)[1]
                    print("\nbackdoor accuracy:", "{0:.2f}".format(bckdr_acc))
            self.triggers.append((K.get_value(K.sigmoid(delta)),K.get_value(K.sigmoid(m)),K.get_value(K.sum(K.abs(K.sigmoid(m))))))


        with open(r'C:\Users\ACER\IdeaProjects\ece5831\triggers.npy', 'wb') as f:
            pickle.dump(self.triggers,f)

    def draw_all_triggers(self):

        if len(self.triggers) != self.num_classes:
            try:
                with open(r'C:\Users\ACER\IdeaProjects\ece5831\triggers.npy', 'rb') as f:
                    self.triggers = pickle.load(f)
            except:
                print("you need to reverse engineer triggers, first.")
                exit()

        plt.cla()
        plt.figure(figsize=(43,43))
        for i in range(self.num_classes):
            ax = plt.subplot(7,7,i+1)
            ax.imshow(self.triggers[i][0]*self.triggers[i][1])
            ax.set_title("Class "+str(i)+" L1:"+"{0:.2f}".format(self.triggers[i][2]),fontsize=36)
            ax.set_xticks([])
            ax.set_yticks([])
        plt.savefig("all_triggers.png")
        plt.close()

    def backdoor_detection(self):

        if len(self.triggers) != self.num_classes:
            try:
                with open(r'C:\Users\ACER\IdeaProjects\ece5831\triggers.npy', 'rb') as f:
                    self.triggers = pickle.load(f)
            except:
                print("you need to reverse engineer triggers, first.")
                exit()

        _,_,l1_norms = zip(*self.triggers)

        stringency = 3.0
        median = np.median(l1_norms,axis=0)
        MAD = 1.4826*np.median(np.abs(l1_norms-median),axis=0)
        in_ditribution = (median-stringency*MAD < l1_norms) * (l1_norms < median+stringency*MAD)
        out_of_distribution = np.logical_not(in_ditribution)
        outliers = np.where(out_of_distribution)[0]

        for possible_target_label in outliers:
            x_samples = []
            for base_label in range(self.num_classes):
                if base_label == possible_target_label:
                    continue
                possible_idx = (np.where(self.Y==base_label)[0]).tolist()
                idx = random.sample(possible_idx,min(self.num_samples_per_label,len(possible_idx)))
                x_samples.append(self.X[idx,::])

            x_samples = np.vstack(x_samples)
            poisoned_x_samples = self.trigger_insert(X=x_samples,Delta=self.triggers[possible_target_label][0], M=self.triggers[possible_target_label][1])
            y_t = (np.ones((poisoned_x_samples.shape[0]))*possible_target_label)
            y_t = keras.utils.to_categorical(y_t,self.num_classes)
            bckdr_acc = self.model.evaluate(poisoned_x_samples,y_t,verbose=0)[1]
            if 0.75 < bckdr_acc:
                print("There is a possible backdoor to label ", possible_target_label, " with ", "{0:.2f}".format(100*bckdr_acc),"% accuracy.")

def main():
    # Define the directory paths for each class
    tawfic_directory = r"C:\Users\ACER\IdeaProjects\ece5831\Images\Tawfic"
    unknown_directory = r"C:\Users\ACER\IdeaProjects\ece5831\Images\Unknown"

    # Initialize lists to store resized images and labels
    x_train = []
    y_train = []

    # Function to resize an image
    def resize_image(image_path, target_size):
        image = Image.open(image_path)
        image_resized = image.resize(target_size)
        return np.array(image_resized)

    # Define the target size for resizing (e.g., (100, 100))
    target_size = (160, 160)

    # Iterate through the images in the tawfic directory
    for filename in os.listdir(tawfic_directory):
        if filename.endswith(".jpg"):
            image_path = os.path.join(tawfic_directory, filename)
            resized_image = resize_image(image_path, target_size)
            x_train.append(resized_image)
            y_train.append('tawfic')

    # Iterate through the images in the unknown directory
    for filename in os.listdir(unknown_directory):
        if filename.endswith(".jpg"):
            image_path = os.path.join(unknown_directory, filename)
            resized_image = resize_image(image_path, target_size)
            x_train.append(resized_image)
            y_train.append('unknown')

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    num_classes = len(np.unique(y_train))
    print("There are", num_classes, "classes in the dataset.")
    x_train = x_train / 255.0
    model = keras.models.load_model(r'C:\Users\ACER\IdeaProjects\ece5831\poisoned_model.h5')
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    NC = NeuralCleanse(X=x_train, Y=y_train_encoded, model=model, num_samples=25)
    NC.draw_all_triggers()
    NC.reverse_engineer_triggers()
    NC.backdoor_detection()

if __name__ == "__main__":
    main()
