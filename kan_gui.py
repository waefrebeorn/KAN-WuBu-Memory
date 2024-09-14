import tkinter as tk
from tkinter import scrolledtext
import threading
import logging
import traceback
from kan_emotional_character_llama_hf import KANEmotionalCharacter  # Updated import

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class KANGUI:
    def __init__(self, master):
        self.master = master
        master.title("KAN Emotional Character Interaction (Llama-3.1-8b-Instruct)")  # Updated title

        self.chat_display = scrolledtext.ScrolledText(master, state='disabled', height=20, width=80)
        self.chat_display.grid(row=0, column=0, columnspan=2, padx=10, pady=10)

        self.input_field = tk.Entry(master, width=70)
        self.input_field.grid(row=1, column=0, padx=10, pady=10)
        self.input_field.bind('<Return>', self.send_message)  # Allow sending with Enter key

        self.send_button = tk.Button(master, text="Send", command=self.send_message)
        self.send_button.grid(row=1, column=1, padx=10, pady=10)

        self.status_label = tk.Label(master, text="Status: Initializing...")
        self.status_label.grid(row=2, column=0, columnspan=2, padx=10, pady=5)

        self.character = None
        self.is_first_message = True

        self.display_message("Initializing KAN Emotional Character (Llama-3.1-8b-Instruct)... Please wait.")
        self.initialize_character()

    def initialize_character(self):
        def init():
            try:
                self.character = KANEmotionalCharacter()
                self.master.after(0, self.display_message, "KAN Emotional Character (Llama-3.1-8b-Instruct) initialized successfully.")
                self.master.after(0, self.display_message, "Please provide the character description in your first message.")
                self.master.after(0, self.display_message, "Type 'exit' to end the conversation.")
                self.master.after(0, self.update_status, "Ready")
            except Exception as e:
                self.master.after(0, self.display_message, f"Error initializing character: {str(e)}")
                self.master.after(0, self.update_status, "Error")
                logging.error(f"Error initializing character: {str(e)}")

        threading.Thread(target=init).start()

    def send_message(self, event=None):
        user_input = self.input_field.get()
        self.input_field.delete(0, tk.END)
        self.display_message(f"You: {user_input}")

        if user_input.lower() == 'exit':
            self.master.quit()
            return

        if self.character is None:
            self.display_message("Character is not initialized yet. Please wait.")
            return

        if self.is_first_message:
            self.character.set_system_prompt(user_input)
            self.display_message("Character description set. You can now start interacting with the character.")
            self.is_first_message = False
        else:
            self.send_button.config(state='disabled')
            self.update_status("Generating response...")
            threading.Thread(target=self.generate_response, args=(user_input,)).start()

    def generate_response(self, user_input):
        try:
            self.update_status("Generating response...")
            response = self.character.generate_response(user_input)
            self.master.after(0, self.display_message, f"Character: {response}")
            self.master.after(0, self.display_message, f"Current Emotion: {self.character.emotional_state.get_emotion()}")
        except Exception as e:
            error_message = f"Error generating response: {str(e)}\n"
            error_message += f"Error type: {type(e).__name__}\n"
            error_message += f"Error details: {traceback.format_exc()}"
            self.master.after(0, self.display_message, error_message)
            logging.error(error_message)
        finally:
            self.master.after(0, self.send_button.config, {'state': 'normal'})
            self.update_status("Ready")

    def update_status(self, status):
        gpu_usage = self.character.get_gpu_usage() if self.character else ""
        self.status_label.config(text=f"Status: {status} - {gpu_usage}")

    def display_message(self, message):
        self.chat_display.configure(state='normal')
        self.chat_display.insert(tk.END, message + "\n")
        self.chat_display.configure(state='disabled')
        self.chat_display.see(tk.END)
        logging.info(message)

        # Update GPU usage in status
        if self.character:
            self.update_status("Ready")

if __name__ == "__main__":
    root = tk.Tk()
    gui = KANGUI(root)
    root.mainloop()