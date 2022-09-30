import tensorflow as tf
import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog
from tkinter.filedialog import asksaveasfile, askopenfilename
import matplotlib.pyplot as plt
import numpy as np
from PIL import ImageTk, Image, ImageDraw
import PIL
from tkinter import *
import cv2
cv2.ocl.setUseOpenCL(False)
cv2.destroyAllWindows()
import os, sys
from collections import Counter
import ctypes
import subprocess
print("tf version : ",tf.__version__)
print("numpy version:",np.__version__)
print("cv2 version:",cv2.__version__) 


def drawing():
    model = tf.keras.models.load_model("./digit_pack.h5")
    width = 300
    height = 300
    center = height//2
    white = (255, 255, 255)
    green = (0,128,0)
    
    
    def predict():
        img=cv2.imread('image.png',0)
        img=cv2.bitwise_not(img)

        img=cv2.resize(img,(28,28))
        img=img.reshape(1,28,28,1)
        img=img.astype('float32')
        img=img/255.0

        pred=model.predict(img)
        myLabel.config(text = "Predicted digit : "+str(pred[0].argmax()))
        
    def load():
        root.filename = filedialog.askopenfilename(initialdir='', title='open_file', filetypes=(
            ('png files', '*.png'), ('jpg files', '*.jpg'), ('all files', '*.*')))
        img_path = ImageTk.PhotoImage(file=r"C:/Users/MSI/AppData/Local/Programs/Python/Python38/Scripts/DataSet/1.png")
        shape = cv.create_image(width/2, height/2, image = img_path)
        
        #global path
        path=root.filename
        global img
        img=cv2.imread(path)
        show_im(img)
        
    def show_im(img) :
        cv2.imshow(" ", img)
        cv.waitKey(0)
        cv2.destroyAllWindows()
                
    def save():
        filename = "image.png"
        image1.save(filename)
        predict()
    
    def save_file():
        files = [('All Files', '*.*'),
                 ('PNG Files', '*.png'),
                 ('JPEG Files', '*.jpeg'), 
                ('Python Files', '*.py'),
                ('Text Document', '*.txt')]
        file = asksaveasfile(filetypes = files, defaultextension = files)
        
    def paint(event):
        x1, y1 = (event.x - 1.5), (event.y - 1.5)
        x2, y2 = (event.x + 1.5), (event.y + 1.5)
        cv.create_oval(x1, y1, x2, y2, fill="black",width=15)
        draw.line([x1, y1, x2, y2],fill="black",width=15)
        

    def clear():
        cv.delete("all")
        myLabel.config(text = "Predicted digit : ")
        plt.close(1)
        
    def setting():
        mb = tk.messagebox.showinfo("Setting Info", "Flatten Layer => 1,                                                     Dense Layer1 => hidden layer node_num : 512,                                      activation_Func : 'relu' ,                             Dropout(0.2) => 1,                                                     Dense Layer2 => output : 10, activation_Func : 'softmax' ")
    
    
    def training():
        subprocess.Popen([sys.executable, "MNIST_1.py"])
    
    def evaluation():
        img=cv2.imread('image.png',0)
        img=cv2.bitwise_not(img)

        img=cv2.resize(img,(28,28))
        img=img.reshape(1,28,28,1)
        img=img.astype('float32')
        img=img/255.0
        #print(img)
        pred=model.predict(img)
        myLabel.config(text = "Predicted digit : "+str(pred[0].argmax()))
        plt.bar(list(range(0,10)) , pred[0])
        plt.xticks(list(range(0,10)))
        plt.show()


    root = tk.Tk()
    root.title("SJ_MNIST.EAZISLOGIC.ver")
    root.geometry("850x650+100+100")
    root.resizable(False, False)
    text = Text(root)
    text.grid()
    frm1 = tk.LabelFrame(root, text = "MNIST", pady=10, padx=15)
    frm1.grid(row=0, column=0, padx=15, pady=15, sticky="nswe")
    
    myLabel = Label(frm1 , text = "    Predicted digit :    " , font = "Times 12 bold")
    myLabel.place(x=20, y= 50, width=550, height=50)
    cv = Canvas(frm1, width=width, height=height, bg='white')
    cv.place()
    
    image1 = PIL.Image.new("RGB", (width, height), white)
    draw = ImageDraw.Draw(image1)
    cv.grid()
    cv.bind("<B1-Motion>", paint)
    
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    
    frm2 = tk.Frame(root, pady=15)
    frm2.grid(row=1, column=0, pady=15)
    
    btn1 = Button(frm2 , text = "prediction", width=12, command = save)
    btn1.pack()
    
    btn2 = Button(frm1 , text = "Clear",width=12, command = clear)
    btn2.place(anchor=S, x=730, y=130)
    
    btn3 = tk.Button(frm1, text= "Save", width=12, command=save_file)
    btn3.place(anchor=S, x=630, y=230)
    
    btn4 = tk.Button(frm1, text= "Load", width=12, command=load)
    btn4.place(anchor=S, x=630, y=130)

    btn5 = tk.Button(frm1, text= "Setting", width=12, command=setting)
    btn5.place(anchor=S, x=730, y=230)
    cv.create_window(300, 300, window=btn5.place())
    
    btn6 = tk.Button(frm1, text= "Training", width=12, command=training)
    btn6.place(anchor=S, x=630, y=330)
    
    btn7 = tk.Button(frm1, text= "Evaluation", width=12, command=evaluation)
    btn7.place(anchor=S, x=730, y=330)
    
    frm1.columnconfigure(0, weight=1)
    frm1.rowconfigure(0, weight=1)

    root.mainloop()

drawing()
