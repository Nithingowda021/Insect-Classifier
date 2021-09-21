import tkinter as tk
from tkinter.filedialog import askopenfilename
from PIL import ImageTk,Image
from tkinter import messagebox
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import model_from_json
from tensorflow.keras.preprocessing.image import ImageDataGenerator



class main_page:
    def __init__ (self, window):
        self.window = window
        self.window.title("Deep Learning based Insect Classifier")
        self.window.geometry("1350x700")
        self.window.resizable(width=False, height=False)


        self.topics_names = ["Pre_process_Dataset", "Train Dataset", "Select Images for Testing"]

        self.btn_commands = [lambda: screen.Pre_process_Dataset(screen.dummy()),
                        lambda: screen.Train_the_complete_dataset(screen.dummy()),
                        lambda: screen.Test_an_Image(screen.dummy())]

        self.image_path = "Images.jpg"


        # ---- Lables -------------------------
        title = tk.Label(self.window, text="Deep Learning based Insect Classification", font=("times new roman", 40, "bold"),
                      bg="#BFCEFF", fg="red", bd=10, relief=tk.GROOVE)
        title.place(x=0, y=0, relwidth=1)


    def open_file(self):
        """Open and select image for processing"""
        filepath = askopenfilename(filetypes=[("Image Files", ("*.jpg", "*.png")), ("All Files", "*.*")])
        if not filepath:
            return
        return filepath

    def dummy(self):
        # frame 1
        frm1 = tk.Frame(self.window, bd=4, relief=tk.RIDGE, bg="white")
        frm1.place(x=25, y=160, width=645, height=510)

        # frame 2
        frm2 = tk.Frame(self.window, bd=4, relief=tk.RIDGE, bg="white")
        frm2.place(x=670, y=160, width=645, height=510)
        return (frm1, frm2)

    def lowerPart(self):
        frm = tk.Frame(self.window, bd=4, relief=tk.RIDGE, bg="#007AA8")
        frm.place(x=20, y=155, width=1300, height=520)

    def options_bar(self):

        frm_options = tk.Frame(self.window, bd=4, relief=tk.RIDGE, bg="#007AA8")
        frm_options.place(x=20, y=100, width=1300, height=50)

        for index, text in enumerate(self.topics_names):

            options_btn = tk.Button(master=frm_options, text=text, width=60, height=2, command=self.btn_commands[index])
            options_btn.grid(row=0, column=index, sticky=tk.NS)


        screen.lowerPart()

    def btn_select_image(self, frm2):
        self.image_path = screen.open_file()
        if self.image_path:
            image_name = self.image_path.split("/")[-1]

            # frm 2
            tk.Label(master=frm2, text=image_name, bg="#007AA8", fg="white",
                     font=("times new roman", 20, "bold")).place(x=0, y=0, relwidth=1)

            # image
            photo = Image.open(self.image_path)
            photo = photo.resize((600, 400), Image.ANTIALIAS)
            photo.save("ArtWrk.ppm")

            img = ImageTk.PhotoImage(Image.open("ArtWrk.ppm"))
            image_frame = tk.Label(frm2, bd=4, image=img)
            image_frame.image = img
            image_frame.place(x=0, y=40)

    def btn_run(self,image_path,index):
        ###if index == 0:


        if index == 2:
            import tensorflow as tf
            classifierLoad = tf.keras.models.load_model('model.h5')

            import numpy as np
            from keras.preprocessing import image
            test_image = image.load_img(self.image_path, target_size=(200, 200))
            test_image = image.img_to_array(test_image)
            test_image = np.expand_dims(test_image, axis=0)
            result = classifierLoad.predict(test_image)


            for i in range(0,1):
                if  result[0][0] == 1:
                    messagebox.showinfo('Result of Insect Classification',
                                        'The result of classification is: Auchenorrhyncha')
                elif  result[0][1] == 1:
                    messagebox.showinfo('Result of Insect Classification',
                                        'The result of classification is: Heteroptera')
                elif result[0][2] == 1:
                    messagebox.showinfo('Result of Insect Classification',
                                        'The result of classification is: Hymenoptera')
                elif  result[0][3] == 1:
                    messagebox.showinfo('Result of Insect Classification',
                                        'The result of classification is: Lepidoptera')
                elif result[0][4] == 1:
                    messagebox.showinfo('Result of Insect Classification',
                                        'The result of classification is: Megalptera')
                elif  result[0][5] == 1:
                    messagebox.showinfo('Result of Insect Classification',
                                        'The result of classification is: Neuroptera')
                elif result[0][6] == 1:
                    messagebox.showinfo('Result of Insect Classification',
                                        'The result of classification is: Odonata')
                elif  result[0][7] == 1:
                    messagebox.showinfo('Result of Insect Classification',
                                        'The result of classification is: Orthoptera')
                elif result[0][8] == 1:
                    messagebox.showinfo('Result of Insect Classification',
                                        'The result of classification is: Diptera')

    def Pre_process_Dataset(self, dummy):
        frm1 = dummy[0]
        frm2 = dummy[1]

        # frm 1
        tk.Label(master=frm1, text="Please Select the folder for pre-processing", bg="#007AA8", fg="white", font=("times new roman", 20)).pack(fill=tk.X)

        # frm 2
        tk.Label(master=frm2, text="Selected folder is pre-processed", bg="#007AA8", fg="white", font=("times new roman", 20, "bold")).pack(fill=tk.X)

        # creating btns
        btns_frame = tk.Frame(master=frm1, relief=tk.RIDGE, bg="white")
        btns_frame.pack(fill=tk.BOTH)

        btn_select = tk.Button(master=btns_frame, text="Select the folder for pre-processing", width=35, height=2,
                               command=lambda: screen.btn_select_image(frm2))
        btn_select.grid(row=0, column=0, sticky=tk.NS, padx=15, pady=10)

        btn_run = tk.Button(master=btns_frame, text="Run", width=25, height=2,
                            command=lambda: screen.btn_run(self.image_path, 0))
        btn_run.grid(row=0, column=1, sticky=tk.NS, padx=15, pady=10)

        # makeup
        makeup_label = tk.Label(master=btns_frame, text="Makeup Selected Image", bg="white", fg="Black", font=("times new roman", 20, "bold"))
        makeup_label.grid(row=1, column =1,sticky=tk.N, padx=15, pady=10)

        # run makeup
        btn_run = tk.Button(master=btns_frame, text="Run", width=25, height=2,
                            command=lambda: screen.btn_run(self.image_path, 7))
        btn_run.grid(row=2, column=1, sticky=tk.NS, padx=15, pady=10)

    def Train_the_complete_dataset(self, dummy):

        batch_size = 32


        train_datagen = ImageDataGenerator(rescale=1 / 255)

        train_generator = train_datagen.flow_from_directory('C:/Users/user/Desktop/Code with GUI/Insects',target_size=(200, 200),batch_size=batch_size,classes=['Auchenorrhyncha', 'Heteroptera', 'Hymenoptera', 'Lepidoptera', 'Megalptera', 'Neuroptera', 'Odonata',
                 'Orthoptera', 'Diptera'],        class_mode='categorical')

        import tensorflow as tf

        model = tf.keras.models.Sequential([

        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(200, 200, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(128, activation='relu'),

        tf.keras.layers.Dense(9, activation='softmax')
    ])

        model.summary()

        from tensorflow.keras.optimizers import RMSprop

        model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(lr=0.001),
                  metrics=['acc'])

        total_sample = train_generator.n

        n_epochs = 5


        history = model.fit_generator(train_generator,steps_per_epoch=int(total_sample / batch_size),epochs=n_epochs,verbose=1)

        model.save('model.h5')
        print("Training of Insect images dataset completed successfully...................")

    def Test_an_Image(self, dummy):
        frm1 = dummy[0]
        frm2 = dummy[1]

        # frm 1
        tk.Label(master=frm1, text="Select an Image for Insect Classification", bg="#007AA8", fg="white", font=("times new roman", 20)).pack(
            fill=tk.X)

        # frm 2
        tk.Label(master=frm2, text="Selected Insect Image", bg="#007AA8", fg="white",
                 font=("times new roman", 20, "bold")).pack(fill=tk.X)

        # creating btns
        btns_frame = tk.Frame(master=frm1, relief=tk.RIDGE, bg="white")
        btns_frame.pack(fill=tk.BOTH)

        options_btn = tk.Button(master=btns_frame, text="Select an Insect Image", width=25, height=2,
                                command=lambda: screen.btn_select_image(frm2))

        options_btn.grid(row=0, column=0, sticky=tk.NS, padx=15, pady=10)

        options_btn = tk.Button(master=btns_frame, text="Run", width=25, height=2,
                                command=lambda: screen.btn_run(self.image_path, 2))
        options_btn.grid(row=0, column=1, sticky=tk.NS, padx=15, pady=10)



window = tk.Tk()
screen = main_page(window)
screen.options_bar()
window.mainloop()

