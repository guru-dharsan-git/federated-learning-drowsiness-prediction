import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import collections
import warnings
warnings.filterwarnings('ignore')


class Client:
    def __init__(self, client_id, data_path, img_size=(150, 150), batch_size=32):
        self.client_id = client_id
        self.data_path = data_path
        self.img_size = img_size
        self.batch_size = batch_size
        self.model = None
        self.train_data = None
        self.val_data = None
        
    def load_and_preprocess_data(self):
        print(f"Loading data for client {self.client_id}...")
        
        
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            validation_split=0.2
        )
        
        
        self.train_data = train_datagen.flow_from_directory(
            self.data_path,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            subset='training'
        )
        
        
        self.val_data = train_datagen.flow_from_directory(
            self.data_path,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            subset='validation'
        )
        
        print(f"Client {self.client_id} data loaded. Training samples: {self.train_data.samples}, Validation samples: {self.val_data.samples}")
        return self.train_data.class_indices
    
    def build_model(self):
        self.model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(*self.img_size, 3)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
    def set_weights(self, weights):
        self.model.set_weights(weights)
        
    def get_weights(self):
        return self.model.get_weights()
    
    def train(self, epochs=5):
        if self.model is None:
            self.build_model()
            
        history = self.model.fit(
            self.train_data,
            steps_per_epoch=self.train_data.samples // self.batch_size,
            epochs=epochs,
            validation_data=self.val_data,
            validation_steps=self.val_data.samples // self.batch_size
        )
        
        val_loss, val_acc = self.model.evaluate(self.val_data)
        print(f"Client {self.client_id} - Validation accuracy: {val_acc:.4f}")
        
        return history, val_acc
    
    def evaluate(self):
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        val_loss, val_acc = self.model.evaluate(self.val_data)
        return val_loss, val_acc



class Server:
    def __init__(self, model_path="federated_model"):
        self.global_model = None
        self.clients = []
        self.model_path = model_path
        self.accuracy_history = []
        
    def build_global_model(self, input_shape):
        self.global_model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])
        
        self.global_model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
    def add_client(self, client):
        self.clients.append(client)
        
    def federated_averaging(self, client_weights, coefficients=None):
        if coefficients is None:
            
            coefficients = [1.0 / len(client_weights) for _ in range(len(client_weights))]
            
        
        avg_weights = [np.zeros_like(w) for w in client_weights[0]]
        
        
        for i, client_w in enumerate(client_weights):
            for j, w in enumerate(client_w):
                avg_weights[j] += coefficients[i] * w
                
        return avg_weights
    
    def train_federated(self, rounds=5, local_epochs=3):
        if not self.clients:
            raise ValueError("No clients added for federated training")
            
        for federated_round in range(1, rounds + 1):
            print(f"\n--- Federated Round {federated_round}/{rounds} ---")
            
            
            global_weights = self.global_model.get_weights()
            
            
            client_weights = []
            accuracies = []
            
            for client in self.clients:
                
                client.set_weights(global_weights)
                
                
                _, val_acc = client.train(epochs=local_epochs)
                
                
                client_weights.append(client.get_weights())
                accuracies.append(val_acc)
                
            
            new_global_weights = self.federated_averaging(client_weights)
            
            
            self.global_model.set_weights(new_global_weights)
            
            
            avg_acc = np.mean(accuracies)
            self.accuracy_history.append(avg_acc)
            print(f"Round {federated_round} - Average validation accuracy: {avg_acc:.4f}")
            
        
        self.save_model()
        
        return self.accuracy_history
    
    def save_model(self):
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        self.global_model.save(os.path.join(self.model_path, "drowsiness_model.h5"))
        print(f"Global model saved to {os.path.join(self.model_path, 'drowsiness_model.h5')}")
        
    def load_model(self):
        model_file = os.path.join(self.model_path, "drowsiness_model.h5")
        if os.path.exists(model_file):
            self.global_model = tf.keras.models.load_model(model_file)
            print(f"Global model loaded from {model_file}")
            return True
        return False
    
    def plot_accuracy_history(self):
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.accuracy_history) + 1), self.accuracy_history, marker='o')
        plt.title('Federated Learning Accuracy Progression')
        plt.xlabel('Federated Rounds')
        plt.ylabel('Average Validation Accuracy')
        plt.grid(True)
        plt.savefig(os.path.join(self.model_path, 'accuracy_progression.png'))
        plt.show()



def split_dataset_for_clients(data_path, num_clients=3):
    """Split the main dataset into chunks for different clients"""
    client_data_paths = []
    
    
    for i in range(num_clients):
        client_dir = f"client_{i+1}_data"
        if not os.path.exists(client_dir):
            os.makedirs(client_dir)
            
            
            for class_name in os.listdir(data_path):
                class_path = os.path.join(data_path, class_name)
                if os.path.isdir(class_path):
                    os.makedirs(os.path.join(client_dir, class_name), exist_ok=True)
                    
        client_data_paths.append(client_dir)
    
    
    for class_name in os.listdir(data_path):
        class_path = os.path.join(data_path, class_name)
        if os.path.isdir(class_path):
            images = [img for img in os.listdir(class_path) if img.endswith(('.jpg', '.jpeg', '.png'))]
            
            
            np.random.shuffle(images)
            
            
            n = len(images)
            client_images = [images[i:n:num_clients] for i in range(num_clients)]
            
            
            for i, client_dir in enumerate(client_data_paths):
                for img in client_images[i]:
                    src = os.path.join(data_path, class_name, img)
                    dst = os.path.join(client_dir, class_name, img)
                    
                    if not os.path.exists(dst):
                        if os.name == 'nt':  
                            import shutil
                            shutil.copy2(src, dst)
                        else:  
                            os.symlink(os.path.abspath(src), dst)
    
    return client_data_paths



def predict_drowsiness(model_path, image_path, img_size=(150, 150)):
    """
    Predict drowsiness from a single image
    """
    model = tf.keras.models.load_model(model_path)
    
    
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=img_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    
    
    prediction = model.predict(img_array)[0][0]
    
    
    if prediction >= 0.5:
        return "Drowsy", prediction
    else:
        return "Alert", 1 - prediction



def main(data_path, num_clients=3, federated_rounds=5, local_epochs=3):
    """
    Main function to run federated learning for drowsiness detection
    """
    print("Starting drowsiness detection using federated learning...")
    
    
    print(f"Splitting dataset into {num_clients} parts...")
    client_data_paths = split_dataset_for_clients(data_path, num_clients)
    
    
    server = Server(model_path="drowsiness_federated_model")
    
    
    clients = []
    for i, client_path in enumerate(client_data_paths):
        client = Client(i+1, client_path)
        class_indices = client.load_and_preprocess_data()
        client.build_model()
        clients.append(client)
        server.add_client(client)
    
    
    server.build_global_model(input_shape=(150, 150, 3))
    
    
    print("\nStarting federated training...")
    accuracy_history = server.train_federated(rounds=federated_rounds, local_epochs=local_epochs)
    
    
    server.plot_accuracy_history()
    
    print("\nFederated learning completed successfully!")
    print(f"Final model saved to {os.path.join(server.model_path, 'drowsiness_model.h5')}")
    
    return server



if __name__ == "__main__":
    
    dataset_path = "dataset"  
    
    
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path '{dataset_path}' does not exist.")
        print("Please make sure your dataset is organized as follows:")
        print("dataset/")
        print("  └── images/")
        print("      ├── drowsy/    (images of drowsy people)")
        print("      └── alert/     (images of alert people)")
        exit(1)
    
    
    server = main(
        data_path=dataset_path,
        num_clients=3,           
        federated_rounds=5,      
        local_epochs=3           
    )
    
    
    
    
    
    
    
    
    