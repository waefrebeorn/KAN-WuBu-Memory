import tkinter as tk
from tkinter import scrolledtext
import threading
import logging
import traceback
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from kan_emotional_character_llama_hf import KANEmotionalCharacter, TimeoutException

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class KANGUI:
    def __init__(self, master):
        self.master = master
        master.title("KAN Emotional Character Interaction (LLaMA 3.1 8B Instruct)")

        # Configure grid layout
        master.grid_rowconfigure(0, weight=1)
        master.grid_columnconfigure(0, weight=1)

        self.chat_display = scrolledtext.ScrolledText(master, state='disabled', height=20, width=80)
        self.chat_display.grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky='nsew')

        self.input_field = tk.Entry(master, width=70)
        self.input_field.grid(row=1, column=0, padx=10, pady=10, sticky='ew')
        self.input_field.bind('<Return>', self.send_message)

        self.send_button = tk.Button(master, text="Send", command=self.send_message)
        self.send_button.grid(row=1, column=1, padx=10, pady=10, sticky='ew')

        self.status_label = tk.Label(master, text="Status: Initializing...")
        self.status_label.grid(row=2, column=0, columnspan=2, padx=10, pady=5, sticky='w')

        self.character = None
        self.is_first_message = True

        self.display_message("Initializing KAN Emotional Character (LLaMA 3.1 8B Instruct)... Please wait.")
        self.initialize_character()

        # Add emotional control buttons
        self.add_emotion_buttons()

    def initialize_character(self):
        def init():
            try:
                max_memory = 6 * 1024 * 1024 * 1024  # 6GB
                timeout_duration = 300  # 5 minutes

                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(KANEmotionalCharacter, max_memory=max_memory)
                    try:
                        self.character = future.result(timeout=timeout_duration)
                        self.master.after(0, self.display_message, "KAN Emotional Character (LLaMA 3.1 8B Instruct) initialized successfully.")
                        self.master.after(0, self.display_message, "Please provide the character description in your first message.")
                        self.master.after(0, self.display_message, "Type 'exit' to end the conversation.")
                        self.master.after(0, self.update_status, "Ready")
                    except TimeoutError:
                        self.master.after(0, self.display_message, "Error: Character initialization timed out after 5 minutes.")
                        self.master.after(0, self.update_status, "Error")
                        logging.error("Character initialization timed out after 5 minutes")
                        future.cancel()
                    except Exception as e:
                        self.master.after(0, self.display_message, f"Error initializing character: {str(e)}")
                        self.master.after(0, self.update_status, "Error")
                        logging.error(f"Error initializing character: {str(e)}")
            except Exception as e:
                self.master.after(0, self.display_message, f"Unexpected error: {str(e)}")
                self.master.after(0, self.update_status, "Error")
                logging.error(f"Unexpected error during initialization: {str(e)}")
                logging.error(traceback.format_exc())

        threading.Thread(target=init, daemon=True).start()

    def add_emotion_buttons(self):
        emotion_frame = tk.Frame(self.master)
        emotion_frame.grid(row=3, column=0, columnspan=2, padx=10, pady=5)

        emotions = ["Happy", "Sad", "Angry", "Excited", "Calm", "Neutral"]
        for emotion in emotions:
            btn = tk.Button(emotion_frame, text=emotion, command=lambda e=emotion: self.set_emotion(e))
            btn.pack(side=tk.LEFT, padx=5)

    def set_emotion(self, emotion):
        if self.character:
            emotion_map = {
                "Happy": [0.5, 0.3],
                "Sad": [-0.5, -0.3],
                "Angry": [-0.3, 0.5],
                "Excited": [0.3, 0.5],
                "Calm": [0.3, -0.3],
                "Neutral": [0, 0]
            }
            self.character.update_emotional_state(emotion_map[emotion])
            self.display_message(f"Emotion set to: {emotion}")
            self.update_status("Ready")
        else:
            self.display_message("Character not initialized. Please wait.")

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
            try:
                self.character.set_system_prompt(user_input)
                self.display_message("Character description set. You can now start interacting with the character.")
                self.is_first_message = False
            except Exception as e:
                error_message = f"Error setting system prompt: {str(e)}\n{traceback.format_exc()}"
                self.display_message(error_message)
                logging.error(error_message)
        else:
            self.send_button.config(state='disabled')
            self.update_status("Generating response...")
            threading.Thread(target=self.generate_response, args=(user_input,), daemon=True).start()

    def generate_response(self, user_input):
        try:
            response = self.character.generate_response(user_input)
            self.master.after(0, self.display_message, f"Character: {response}")
            self.master.after(0, self.display_message, f"Current Emotion: {self.character.emotional_state.get_emotion()}")
            self.master.after(0, self.get_user_feedback)
        except TimeoutException:
            self.master.after(0, self.display_message, "Error: Response generation timed out.")
            logging.error("Response generation timed out.")
        except Exception as e:
            error_message = f"Error generating response: {str(e)}\n{traceback.format_exc()}"
            self.master.after(0, self.display_message, error_message)
            logging.error(error_message)
        finally:
            self.master.after(0, lambda: self.send_button.config(state='normal'))
            self.master.after(0, self.update_status, "Ready")

    def get_user_feedback(self):
        feedback_window = tk.Toplevel(self.master)
        feedback_window.title("Provide Feedback")

        tk.Label(feedback_window, text="How well did the response match your intent?").pack(pady=10)
        intent_scale = tk.Scale(feedback_window, from_=0, to=1, resolution=0.1, orient=tk.HORIZONTAL)
        intent_scale.pack()

        tk.Label(feedback_window, text="How did the response make you feel?").pack(pady=10)
        emotion_var = tk.StringVar(value="Neutral")
        emotions = ["Happy", "Sad", "Angry", "Excited", "Calm", "Neutral"]
        for emotion in emotions:
            tk.Radiobutton(feedback_window, text=emotion, variable=emotion_var, value=emotion).pack(anchor='w')

        def submit_feedback():
            intent_feedback = intent_scale.get()
            emotion_feedback = self.map_emotion_to_vector(emotion_var.get())
            self.character.update_kan(intent_feedback, emotion_feedback)
            self.display_message(f"Feedback applied. Intent: {intent_feedback}, Emotion: {emotion_var.get()}")
            feedback_window.destroy()

        tk.Button(feedback_window, text="Submit Feedback", command=submit_feedback).pack(pady=10)

    def map_emotion_to_vector(self, emotion):
        emotion_map = {
            "Happy": [0.5, 0.3],
            "Sad": [-0.5, -0.3],
            "Angry": [-0.3, 0.5],
            "Excited": [0.3, 0.5],
            "Calm": [0.3, -0.3],
            "Neutral": [0, 0]
        }
        return emotion_map.get(emotion, [0, 0])

    def update_status(self, status):
        if self.character:
            emotion = self.character.emotional_state.get_emotion()
            emotion_icon = self.get_emotion_icon(emotion)
            color = self.get_emotion_color(emotion)
            self.status_label.config(text=f"Status: {status} {emotion_icon}", fg=color)
        else:
            self.status_label.config(text=f"Status: {status}")

    def get_emotion_icon(self, emotion):
        icons = {
            "Happy": "😊",
            "Sad": "😢",
            "Angry": "😠",
            "Excited": "🤩",
            "Calm": "😌",
            "Neutral": "😐",
        }
        return icons.get(emotion, "❓")

    def get_emotion_color(self, emotion):
        colors = {
            "Happy": "green",
            "Sad": "blue",
            "Angry": "red",
            "Excited": "orange",
            "Calm": "purple",
            "Neutral": "black",
        }
        return colors.get(emotion, "black")

    def display_message(self, message):
        self.chat_display.configure(state='normal')
        self.chat_display.insert(tk.END, message + "\n")
        self.chat_display.configure(state='disabled')
        self.chat_display.see(tk.END)
        logging.info(message)

        if self.character:
            current_status = self.status_label.cget("text")
            if current_status.startswith("Ready"):
                self.update_status("Ready")

if __name__ == "__main__":
    root = tk.Tk()
    gui = KANGUI(root)
    root.mainloop()