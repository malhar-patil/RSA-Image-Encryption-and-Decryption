# Import dependencies
import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog
import os
from questionary import select
from dotenv import load_dotenv
load_dotenv()
import random
import cv2
from datetime import datetime
import pickle
import numpy as np
from PIL import Image, ImageTk

# Set appearence and theme of the app
ctk.set_appearance_mode("Dark")  # Modes: "System" (standard), "Dark", "Light"
ctk.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue" 

# Declare global variables
filePath=" "
pkl_filePath=" "
EValue=0
NValue=0
DValue_entry_value=0
NValue_entry_value=0

# This function calculates (x^y) % p using modular exponentiation
def power(x, y, p):
    result = 1
    x = x % p

    while y > 0:
        # If y is odd, multiply x with result
        if y % 2 == 1:
            result = (result * x) % p
        # y must be even now
        y = y // 2
        x = (x * x) % p

    return result

# This function performs the Miller-Rabin primality test with k rounds of testing
def miller_rabin_test(n, k=5):
    # Base cases
    if n == 2 or n == 3:
        return True
    if n <= 1 or n % 2 == 0:
        return False

    # Write n as (2^r) * d + 1
    d = n - 1
    r = 0
    while d % 2 == 0:
        r += 1
        d //= 2

    # Witness loop
    for _ in range(k):
        a = random.randint(2, n - 2)
        x = power(a, d, n)

        if x == 1 or x == n - 1:
            continue

        # Repeat squaring
        for _ in range(r - 1):
            x = power(x, 2, n)
            if x == n - 1:
                break
        else:
            # If no break occurred in the loop, n is composite
            return False

    # If the test passes for all witnesses, n is probably prime
    return True

#Function to generate random numbers
def generate_random_number(bits):
    num = random.getrandbits(bits)
    
    while num % 2 == 0 or miller_rabin_test(num) != True:
        num = random.getrandbits(bits)

    return num

# This function generates key pairs
def generate_key_pairs(EValue_label,NValue_label,info_label):
    global EValue,NValue
    p = generate_random_number(8)
    q = generate_random_number(8)
    while p == q:
        q = generate_random_number(8)
    
    # Calculate n
    n = p * q

    # Calculate Euler's totient function
    euler_totient = (p - 1) * (q - 1)

    # Calculate e
    e = random.randint(1, euler_totient)
    while e % 2 == 0 or miller_rabin_test(e) != True:
        e = random.randint(1, euler_totient)
    
    # Calculate d
    d = pow(e, -1, euler_totient)

    # Storing public and private keys
    with open(".env", "w") as env_file:
        env_file.write(f"PUBLIC_KEY_E={e}\n")
        env_file.write(f"PUBLIC_KEY_N={n}\n")
        env_file.write(f"PRIVATE_KEY_D={d}\n")
        env_file.write(f"PRIVATE_KEY_N={n}\n")

    EValue=e
    NValue=n
    EValue_label.configure(text=EValue)
    NValue_label.configure(text=NValue)
    info_label.configure(text="Key Pair Generated")


# This function is used to load image files    
def open_image(info_label):
    global filePath
    file_path = filedialog.askopenfilename(
        initialdir="D:/Users/Malhar/Pictures",  # Set the initial directory
        title="Select an Image File",
        filetypes=[("Image files", "*.png;*.jpg;*.jpeg")]  # Specify allowed file types
    )
    info_label.configure(text=(f"File Path: ",file_path))
    filePath=file_path

# This function is used to load .pkl files 
def open_pkl(info_label):
    global pkl_filePath
    file_path = filedialog.askopenfilename(
        initialdir="D:/Users/Malhar/Pictures",  # Set the initial directory
        title="Select a .pkl File",
        filetypes=[("pkl file", "*.pkl")]  # Specify allowed file types
    )
    info_label.configure(text=(f"File Path: ",file_path))
    pkl_filePath=file_path

# Function to encrypt the image
def encrypt():
        global filePath
        img_path=filePath
        img = cv2.imread(img_path)
        height, width, _ = img.shape

        row,col=img.shape[0],img.shape[1]
        enc = [[0 for x in range(col)] for y in range(row)]

        if not os.path.isfile(".env"):
            print("Please generate key pairs first.")
        else:
            # Load keys from .env file
            public_key_e = int(os.getenv("PUBLIC_KEY_E"))
            public_key_n = int(os.getenv("PUBLIC_KEY_N"))

            # Perform encryption
            height, width, _ = img.shape
            for i in range(height):
                for j in range(width):
                    r, g, b = img[i, j]
                    c1 = power(r, public_key_e, public_key_n)
                    c2 = power(g, public_key_e, public_key_n)
                    c3 = power(b, public_key_e, public_key_n)
                    img[i, j] = [c1 % 256, c2 % 256, c3 % 256]

                    # Save encrypted pixel values to the "enc" array
                    enc[i][j] = [c1, c2, c3]

            with open("enc_data.pkl", "wb") as enc_file:
                pickle.dump(enc, enc_file)

            cv2.imwrite("encrypted_image.jpg", img)

# Function to decrypt the image
def decrypt():
        if not os.path.isfile(".env"):
            print("Please generate key pairs first.")
        else:
            # Load keys from .env file
            private_key_d = int(os.getenv("PRIVATE_KEY_D"))
            private_key_n = int(os.getenv("PRIVATE_KEY_N"))

            with open(pkl_filePath, "rb") as enc_file:
                enc = pickle.load(enc_file)

            # Load the encrypted image
            #img = cv2.imread(img_path)
            
            height, width = len(enc), len(enc[0])
            img = np.zeros((height, width, 3), dtype=np.uint8)

            # Perform decryption
            height, width, _ = img.shape
            for i in range(height):
                for j in range(width):
                    c1, c2, c3 = enc[i][j]  # Load encrypted pixel values from the "enc" array
                    m1 = power(c1, private_key_d, private_key_n)
                    m2 = power(c2, private_key_d, private_key_n)
                    m3 = power(c3, private_key_d, private_key_n)
                    img[i, j] = [m1 % 256, m2 % 256, m3 % 256]

            cv2.imwrite("decrypted_image.jpg", img)

# Upsert private key values into .env file
def update_dec(e,n):
    global DValue_entry_value, NValue_entry_value
    DValue_entry_value=e
    NValue_entry_value=n
    with open(".env", "w") as env_file:
        env_file.write(f"PRIVATE_KEY_D={DValue_entry_value}\n")
        env_file.write(f"PRIVATE_KEY_N={NValue_entry_value}\n")        

# decrypt key window
def image_dec_window():
    # setup grid
    app.grid_columnconfigure((0,1,2,3),weight=1,uniform="a")
    app.grid_rowconfigure((0,1,2,3),weight=1,uniform="a")

    # sidebar
    sidebar_frame = ctk.CTkFrame(app,corner_radius=10,fg_color="#343434")
    sidebar_frame.grid(row=0, column=0, rowspan=4,columnspan=1, sticky="nsew",padx=10,pady=10)
    sidebar_frame.grid_rowconfigure((0,1,2,3,4,5),weight=1,uniform="a")
    sidebar_frame.grid_columnconfigure(0,weight=1,uniform="a")

    # sidebar options

    #button1
    button1 = ctk.CTkButton(sidebar_frame,text="Generate key Pairs",corner_radius=10,font=("Helvetica",16,"bold"),command=lambda:load_frame1())
    button1.grid(row=0,column=0,sticky="nsew",padx=5,pady=(0,10))

    #button2
    button2 = ctk.CTkButton(sidebar_frame,text="Encrypt Image",corner_radius=10,font=("Helvetica",16,"bold"),command=lambda:image_enc_window())
    button2.grid(row=1,column=0,sticky="nsew",padx=5,pady=(0,10))

    #button3
    button3 = ctk.CTkButton(sidebar_frame,text="Decrypt Image",corner_radius=10,font=("Helvetica",16,"bold"),command=lambda:image_dec_window())
    button3.grid(row=2,column=0,sticky="nsew",padx=5,pady=(0,10))

    label1 = ctk.CTkLabel(app, text="Decrypt Image", fg_color="transparent",font=("Helvetica",24,"bold"))
    label1.grid(row=0,column=1,rowspan=1,columnspan=3,sticky="nsew",pady=10)

    #Main window middle frame
    middle_frame = ctk.CTkFrame(app,corner_radius=10,fg_color="#343434")
    middle_frame.grid(row=1, column=1, rowspan=2,columnspan=1, sticky="nsew",padx=10,pady=10)
    middle_frame.grid_rowconfigure((0,1,2,3,4),weight=1)
    middle_frame.grid_columnconfigure(0,weight=1,uniform="a")

    EValue_entry = ctk.CTkEntry(middle_frame, placeholder_text="Enter private key (D)")
    EValue_entry.grid(row=0, column=0, rowspan=1,columnspan=1, sticky="ew",padx=10,pady=10)

    NValue_entry = ctk.CTkEntry(middle_frame, placeholder_text="Enter private key (N)")
    NValue_entry.grid(row=1, column=0, rowspan=1,columnspan=1, sticky="ew",padx=10,pady=10)

    button5 = ctk.CTkButton(middle_frame,text="Submit",corner_radius=10,font=("Helvetica",16,"bold"),command=lambda:update_dec(EValue_entry.get(),NValue_entry.get()))
    button5.grid(row=2,column=0,rowspan=1, columnspan=1, sticky="nsew",padx=10,pady=10)



    choose_file_button = tk.Button(
    master=middle_frame,
    text="Choose a .pkl File",
    font=("Helvetica",16,"bold"),
    command=lambda:open_pkl(info_label)  # Bind the open_image function to button click
    )
    choose_file_button.grid(row=4, column=0, rowspan=1, columnspan=1, sticky="nsew", padx=10, pady=10)

    #button4
    button4 = ctk.CTkButton(middle_frame,text="Decrypt Image",corner_radius=10,font=("Helvetica",16,"bold"),command=lambda:decrypt())
    button4.grid(row=5,column=0,rowspan=1, columnspan=1, sticky="nsew",padx=10,pady=10)


    # Image window
    image_frame = ctk.CTkFrame(app,corner_radius=5,fg_color="#343434")
    image_frame.grid(row=1, column=2, rowspan=2,columnspan=2, sticky="nsew",padx=10,pady=10)

    #Information window
    info_frame = ctk.CTkFrame(app,corner_radius=10,fg_color="#343434")
    info_frame.grid(row=3, column=1, rowspan=1,columnspan=3, sticky="nsew",padx=10,pady=10)

    info_label = ctk.CTkLabel(info_frame, text=" ", font=("Helvetica", 12), corner_radius=10, text_color="white", justify="left")
    info_label.grid(row=3, column=1, rowspan=1, columnspan=3, sticky="nsew",padx=10,pady=10)


# encrypt image window
def image_enc_window():
    # setup grid
    app.grid_columnconfigure((0,1,2,3),weight=1,uniform="a")
    app.grid_rowconfigure((0,1,2,3),weight=1,uniform="a")

    # sidebar
    sidebar_frame = ctk.CTkFrame(app,corner_radius=10,fg_color="#343434")
    sidebar_frame.grid(row=0, column=0, rowspan=4,columnspan=1, sticky="nsew",padx=10,pady=10)
    sidebar_frame.grid_rowconfigure((0,1,2,3,4,5),weight=1,uniform="a")
    sidebar_frame.grid_columnconfigure(0,weight=1,uniform="a")

    # sidebar options

    #button1
    button1 = ctk.CTkButton(sidebar_frame,text="Generate key Pairs",corner_radius=10,font=("Helvetica",16,"bold"),command=lambda:load_frame1())
    button1.grid(row=0,column=0,sticky="nsew",padx=5,pady=(0,10))

    #button2
    button2 = ctk.CTkButton(sidebar_frame,text="Encrypt Image",corner_radius=10,font=("Helvetica",16,"bold"),command=lambda:image_enc_window())
    button2.grid(row=1,column=0,sticky="nsew",padx=5,pady=(0,10))

    #button3
    button3 = ctk.CTkButton(sidebar_frame,text="Decrypt Image",corner_radius=10,font=("Helvetica",16,"bold"),command=lambda:image_dec_window())
    button3.grid(row=2,column=0,sticky="nsew",padx=5,pady=(0,10))

    label1 = ctk.CTkLabel(app, text="Encrypt Image", fg_color="transparent",font=("Helvetica",24,"bold"))
    label1.grid(row=0,column=1,rowspan=1,columnspan=3,sticky="nsew",pady=10)

    #Main window middle frame
    middle_frame = ctk.CTkFrame(app,corner_radius=10,fg_color="#343434")
    middle_frame.grid(row=1, column=1, rowspan=2,columnspan=1, sticky="nsew",padx=10,pady=10)
    middle_frame.grid_rowconfigure((0,1,2,3,4),weight=1)
    middle_frame.grid_columnconfigure(0,weight=1,uniform="a")

    pubkeyE_label=ctk.CTkLabel(middle_frame,text="Public Key (E):",font=("Helvetica", 14, "bold"),text_color="white",justify="left")
    pubkeyE_label.grid(row=0,column=0,rowspan=1,columnspan=1,sticky="sw",padx=10,pady=10)

    EValue_frame = ctk.CTkFrame(middle_frame,corner_radius=10,fg_color="black")
    EValue_frame.grid(row=1, column=0, rowspan=1,columnspan=1, sticky="nsew",padx=10,pady=10)

    EValue_label = ctk.CTkLabel(EValue_frame, text=EValue, font=("Helvetica", 14), text_color="white",justify="left")
    EValue_label.grid(row=1, column=0, rowspan=1, columnspan=1, sticky="nw",padx=10,pady=10)

    pubkeyN_label=ctk.CTkLabel(middle_frame,text="Public Key (N):",font=("Helvetica", 14, "bold"),text_color="white",justify="left")
    pubkeyN_label.grid(row=2,column=0,rowspan=1,columnspan=1,sticky="sw",padx=10,pady=10)

    NValue_frame = ctk.CTkFrame(middle_frame,corner_radius=10,fg_color="black")
    NValue_frame.grid(row=3, column=0, rowspan=1,columnspan=1, sticky="nsew",padx=10,pady=10)

    NValue_label = ctk.CTkLabel(NValue_frame, text=NValue, font=("Helvetica", 14), text_color="white",justify="left")
    NValue_label.grid(row=3, column=0, rowspan=1, columnspan=1, sticky="nw",padx=10,pady=10)

    choose_file_button = tk.Button(
    master=middle_frame,
    text="Choose a File",
    font=("Helvetica",16,"bold"),
    command=lambda:display_image(open_image(info_label))  # Bind the open_image function to button click
    )
    choose_file_button.grid(row=4, column=0, rowspan=1, columnspan=1, sticky="nsew", padx=10, pady=10)

    #button4
    button4 = ctk.CTkButton(middle_frame,text="Encrypt Image",corner_radius=10,font=("Helvetica",16,"bold"),command=lambda:encrypt())
    button4.grid(row=5,column=0,rowspan=1, columnspan=1, sticky="nsew",padx=10,pady=10)


    # Image window
    image_frame = ctk.CTkFrame(app,corner_radius=5,fg_color="#343434")
    image_frame.grid(row=1, column=2, rowspan=2,columnspan=2, sticky="nsew",padx=10,pady=10)
    image_frame.grid_columnconfigure(0,weight=1)
    image_frame.grid_rowconfigure(0,weight=1)

    def display_image(img_path):
        # Load the image
        image = Image.open(filePath)

        # Resize the image to fit the window
        image = image.resize((400,400), Image.BICUBIC)

        # Convert the Image object to a PhotoImage object
        photo_image = ImageTk.PhotoImage(image)

        # Display the image in a Label widget
        image_label = tk.Label(image_frame, image=photo_image,bg="#343434")
        image_label.image = photo_image  # Keep a reference to the image to prevent it from being garbage collected
        image_label.grid(row=0, column=0,sticky="nsew", padx=10, pady=10)

    #Information window
    info_frame = ctk.CTkFrame(app,corner_radius=10,fg_color="#343434")
    info_frame.grid(row=3, column=1, rowspan=1,columnspan=3, sticky="nsew",padx=10,pady=10)

    info_label = ctk.CTkLabel(info_frame, text=" ", font=("Helvetica", 12), corner_radius=10, text_color="white", justify="left")
    info_label.grid(row=3, column=1, rowspan=1, columnspan=3, sticky="nsew",padx=10,pady=10)


   
# Key generation window
def load_frame1():
    # setup grid
    app.grid_columnconfigure((0,1,2,3),weight=1,uniform="a")
    app.grid_rowconfigure((0,1,2,3),weight=1,uniform="a")

    # sidebar
    sidebar_frame = ctk.CTkFrame(app,corner_radius=10,fg_color="#343434")
    sidebar_frame.grid(row=0, column=0, rowspan=4,columnspan=1, sticky="nsew",padx=10,pady=10)
    sidebar_frame.grid_rowconfigure((0,1,2,3,4,5),weight=1,uniform="a")
    sidebar_frame.grid_columnconfigure(0,weight=1,uniform="a")
    # Image window
    image_frame = ctk.CTkFrame(app,corner_radius=5,fg_color="#343434")
    image_frame.grid(row=1, column=2, rowspan=2,columnspan=2, sticky="nsew",padx=10,pady=10)
    # sidebar options

    #button1
    button1 = ctk.CTkButton(sidebar_frame,text="Generate key Pairs",corner_radius=10,font=("Helvetica",16,"bold"),command=lambda:load_frame1())
    button1.grid(row=0,column=0,sticky="nsew",padx=5,pady=(0,10))

    #button2
    button2 = ctk.CTkButton(sidebar_frame,text="Encrypt Image",corner_radius=10,font=("Helvetica",16,"bold"),command=lambda:image_enc_window())
    button2.grid(row=1,column=0,sticky="nsew",padx=5,pady=(0,10))

    #button3
    button3 = ctk.CTkButton(sidebar_frame,text="Decrypt Image",corner_radius=10,font=("Helvetica",16,"bold"),command=lambda:image_dec_window())
    button3.grid(row=2,column=0,sticky="nsew",padx=5,pady=(0,10))

    label1 = ctk.CTkLabel(app, text="Generate Key Pairs", fg_color="transparent",font=("Helvetica",24,"bold"))
    label1.grid(row=0,column=1,rowspan=1,columnspan=3,sticky="nsew",pady=10)

    #Main window middle frame
    middle_frame = ctk.CTkFrame(app,corner_radius=10,fg_color="#343434")
    middle_frame.grid(row=1, column=1, rowspan=2,columnspan=1, sticky="nsew",padx=10,pady=10)
    middle_frame.grid_rowconfigure((0,1,2,3,4),weight=1,uniform="a")
    middle_frame.grid_columnconfigure(0,weight=1,uniform="a")

    pubkeyE_label=ctk.CTkLabel(middle_frame,text="Public Key (E):",font=("Helvetica", 14, "bold"),text_color="white",justify="left")
    pubkeyE_label.grid(row=0,column=0,rowspan=1,columnspan=1,sticky="sw",padx=10,pady=10)

    EValue_frame = ctk.CTkFrame(middle_frame,corner_radius=10,fg_color="black")
    EValue_frame.grid(row=1, column=0, rowspan=1,columnspan=1, sticky="nsew",padx=10,pady=10)

    EValue_label = ctk.CTkLabel(EValue_frame, text=" ", font=("Helvetica", 14,"bold"), text_color="white",justify="left")
    EValue_label.grid(row=0, column=0, rowspan=1, columnspan=1, sticky="nw",padx=10,pady=10)

    pubkeyN_label=ctk.CTkLabel(middle_frame,text="Public Key (N):",font=("Helvetica", 14, "bold"),text_color="white",justify="left")
    pubkeyN_label.grid(row=2,column=0,rowspan=1,columnspan=1,sticky="sw",padx=10,pady=10)

    NValue_frame = ctk.CTkFrame(middle_frame,corner_radius=10,fg_color="black")
    NValue_frame.grid(row=3, column=0, rowspan=1,columnspan=1, sticky="nsew",padx=10,pady=10)

    NValue_label = ctk.CTkLabel(NValue_frame, text=" ", font=("Helvetica", 14), text_color="white",justify="left")
    NValue_label.grid(row=0, column=0, rowspan=1, columnspan=1, sticky="nw",padx=10,pady=10)

    button4 = ctk.CTkButton(middle_frame,text="Generate Keys",corner_radius=10,font=("Helvetica",14,"bold"),command=lambda:generate_key_pairs(EValue_label,NValue_label,info_label))
    button4.grid(row=4,column=0,sticky="nsew",padx=10,pady=10)

    # Image window
    image_frame = ctk.CTkFrame(app,corner_radius=5,fg_color="#343434")
    image_frame.grid(row=1, column=2, rowspan=2,columnspan=2, sticky="nsew",padx=10,pady=10)

    #Information window
    info_frame = ctk.CTkFrame(app,corner_radius=10,fg_color="#343434")
    info_frame.grid(row=3, column=1, rowspan=1,columnspan=3, sticky="nsew",padx=10,pady=10)

    info_label = ctk.CTkLabel(info_frame, text=" ", font=("Helvetica", 12), corner_radius=10, text_color="white", justify="left")
    info_label.grid(row=3, column=1, rowspan=1, columnspan=3, sticky="nsew",padx=10,pady=10)


#Default Window
#initialize the window
app=ctk.CTk()
app.title("Image Encryption and Decryption Tool")
app.geometry("800x600")
# setup grid
app.grid_columnconfigure((0,1,2,3),weight=1,uniform="a")
app.grid_rowconfigure((0,1,2,3),weight=1,uniform="a")
# sidebar
sidebar_frame = ctk.CTkFrame(app,corner_radius=10,fg_color="#343434")
sidebar_frame.grid(row=0, column=0, rowspan=4,columnspan=1, sticky="nsew",padx=10,pady=10)
sidebar_frame.grid_rowconfigure((0,1,2,3,4,5),weight=1,uniform="a")
sidebar_frame.grid_columnconfigure(0,weight=1,uniform="a")
# Image window
image_frame = ctk.CTkFrame(app,corner_radius=5,fg_color="#343434")
image_frame.grid(row=1, column=2, rowspan=2,columnspan=2, sticky="nsew",padx=10,pady=10)

#Information window
info_frame = ctk.CTkFrame(app,corner_radius=10,fg_color="#343434")
info_frame.grid(row=3, column=1, rowspan=1,columnspan=3, sticky="nsew",padx=10,pady=10)

info_label = ctk.CTkLabel(info_frame, text=" ", font=("Helvetica",14), corner_radius=10, text_color="white", justify="left")
info_label.grid(row=3, column=1, rowspan=1, columnspan=3, sticky="nsew",padx=10,pady=10)

# sidebar options

#button1
button1 = ctk.CTkButton(sidebar_frame,text="Generate key Pairs",corner_radius=10,font=("Helvetica",16,"bold"),command=lambda:load_frame1())
button1.grid(row=0,column=0,sticky="nsew",padx=5,pady=(0,10))

#button2
button2 = ctk.CTkButton(sidebar_frame,text="Encrypt Image",corner_radius=10,font=("Helvetica",16,"bold"),command=lambda:image_enc_window())
button2.grid(row=1,column=0,sticky="nsew",padx=5,pady=(0,10))

#button3
button3 = ctk.CTkButton(sidebar_frame,text="Decrypt Image",corner_radius=10,font=("Helvetica",16,"bold"),command=lambda:image_dec_window())
button3.grid(row=2,column=0,sticky="nsew",padx=5,pady=(0,10))

label1 = ctk.CTkLabel(app, text="Image Encryption and Decryption Tool", fg_color="transparent",font=("Helvetica",24,"bold"))
label1.grid(row=0,column=1,rowspan=1,columnspan=3,sticky="nsew",pady=10)

#Main window middle frame
middle_frame = ctk.CTkFrame(app,corner_radius=10,fg_color="#343434")
middle_frame.grid(row=1, column=1, rowspan=2,columnspan=1, sticky="nsew",padx=10,pady=10)
middle_frame.grid_rowconfigure((0,1,2,3,4),weight=1,uniform="a")
middle_frame.grid_columnconfigure(0,weight=1,uniform="a")
# Image window
image_frame = ctk.CTkFrame(app,corner_radius=5,fg_color="#343434")
image_frame.grid(row=1, column=2, rowspan=2,columnspan=2, sticky="nsew",padx=10,pady=10) 

#Information window
info_frame = ctk.CTkFrame(app,corner_radius=10,fg_color="#343434")
info_frame.grid(row=3, column=1, rowspan=1,columnspan=3, sticky="nsew",padx=10,pady=10)

info_label = ctk.CTkLabel(info_frame, text=" ", font=("Helvetica", 12), corner_radius=10, text_color="white", justify="left")
info_label.grid(row=3, column=1, rowspan=1, columnspan=3, sticky="nsew",padx=10,pady=10)


app.mainloop()
