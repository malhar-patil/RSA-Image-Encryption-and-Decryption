#import required libraries/packages
import os
from questionary import select
from dotenv import load_dotenv
load_dotenv()
import random
import cv2
from datetime import datetime
import pickle 
import numpy as np

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
def generate_key_pairs():
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

if __name__ == "__main__":
    img_path="D:\\Users\\Malhar\\Downloads\\modestas-urbonas-vj_9l20fzj0-unsplash.jpg"
    # Load the image
    img = cv2.imread(img_path)

    height, width, _ = img.shape
    print('Image Dimensions:')
    print('Height:', height, 'px')
    print('Width:', width, 'px')

    #Create a list- enc 
    row,col=img.shape[0],img.shape[1]
    enc = [[0 for x in range(col)] for y in range(row)]

    options_ask = ["Generate key pairs", "Encrypt Image", "Decrypt Image", "Exit"]
    scan = select("What do you want to do today?", choices=options_ask).ask()

    if scan == "Generate key pairs":
        generate_key_pairs()
        print("Key pairs generated successfully!")

    elif scan == "Encrypt Image":
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

            # Display th image
            cv2.imshow('Image', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            # Upsert enc into enc_data.pkl file
            with open("enc_data.pkl", "wb") as enc_file:
                pickle.dump(enc, enc_file)

            # Save the image
            cv2.imwrite("encrypted_image.jpg", img)

    elif scan == "Decrypt Image":
        if not os.path.isfile(".env"):
            print("Please generate key pairs first.")
        else:
            # Load keys from .env file
            private_key_d = int(os.getenv("PRIVATE_KEY_D"))
            private_key_n = int(os.getenv("PRIVATE_KEY_N"))

            #Load contents of .pkl file
            with open("enc_data.pkl", "rb") as enc_file:
                enc = pickle.load(enc_file)
            
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

            
            # Display the image
            cv2.imshow('Image', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            # Save the decrypted image
            cv2.imwrite("decrypted_image.jpg", img)


    else:
        print("Press Enter to exit")