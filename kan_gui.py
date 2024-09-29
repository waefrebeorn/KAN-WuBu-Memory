import tkinter as tk
from tkinter import scrolledtext, messagebox, ttk, filedialog
import threading
import logging
import traceback
from llama_32_1b_tool import LLaMA32TensorRTTool
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LLAMA32GUI:
    def __init__(self, master):
        self.master = master
        master.title("LLaMA 3.2 1B Instruct KAN Interaction")

        # Create notebook for tabs
        self.notebook = ttk.Notebook(master)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Create main tab
        self.main_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.main_tab, text="Main")

        # Create graphs tab
        self.graphs_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.graphs_tab, text="Graphs")

        # Main tab layout
        self.setup_main_tab()

        # Graphs tab layout
        self.setup_graphs_tab()

        self.llama_tool = None
        self.is_first_message = True

        self.initialize_tool()

    def setup_main_tab(self):
        self.chat_display = scrolledtext.ScrolledText(self.main_tab, state='disabled', height=20, width=80)
        self.chat_display.grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky='nsew')

        self.input_field = tk.Entry(self.main_tab, width=70)
        self.input_field.grid(row=1, column=0, padx=10, pady=10, sticky='ew')
        self.input_field.bind('<Return>', self.send_message)

        self.send_button = tk.Button(self.main_tab, text="Send", command=self.send_message)
        self.send_button.grid(row=1, column=1, padx=10, pady=10, sticky='ew')

        self.status_label = tk.Label(self.main_tab, text="Status: Initializing...")
        self.status_label.grid(row=2, column=0, columnspan=2, padx=10, pady=5, sticky='w')

        self.time_label = tk.Label(self.main_tab, text="Current Time: N/A")
        self.time_label.grid(row=3, column=0, columnspan=2, padx=10, pady=5, sticky='w')

        self.sleep_button = tk.Button(self.main_tab, text="Sleep", command=self.sleep_kan, state='disabled')
        self.sleep_button.grid(row=4, column=0, padx=10, pady=5, sticky='w')

        self.save_state_button = tk.Button(self.main_tab, text="Save KAN State", command=self.save_kan_state, state='disabled')
        self.save_state_button.grid(row=4, column=1, padx=10, pady=5, sticky='e')

        self.emotion_label = tk.Label(self.main_tab, text="Emotion: N/A")
        self.emotion_label.grid(row=5, column=0, columnspan=2, padx=10, pady=5, sticky='w')

        self.emotion_feedback_label = tk.Label(self.main_tab, text="Emotional Feedback:")
        self.emotion_feedback_label.grid(row=6, column=0, padx=10, pady=5, sticky='w')

        self.emotion_feedback_entry = tk.Entry(self.main_tab, width=20)
        self.emotion_feedback_entry.grid(row=6, column=1, padx=10, pady=5, sticky='ew')

        self.compliance_label = tk.Label(self.main_tab, text="Compliance Rating (0-1):")
        self.compliance_label.grid(row=7, column=0, padx=10, pady=5, sticky='w')

        self.compliance_entry = tk.Entry(self.main_tab, width=10)
        self.compliance_entry.grid(row=7, column=1, padx=10, pady=5, sticky='ew')

        self.feedback_button = tk.Button(self.main_tab, text="Submit Feedback", command=self.submit_feedback, state='disabled')
        self.feedback_button.grid(row=8, column=0, columnspan=2, padx=10, pady=5)

        self.load_state_button = tk.Button(self.main_tab, text="Load Saved State", command=self.load_saved_state)
        self.load_state_button.grid(row=9, column=0, padx=10, pady=5, sticky='w')

        self.new_conversation_button = tk.Button(self.main_tab, text="Start New Conversation", command=self.start_new_conversation)
        self.new_conversation_button.grid(row=9, column=1, padx=10, pady=5, sticky='e')

    def setup_graphs_tab(self):
        self.fig, self.axes = plt.subplots(3, 2, figsize=(10, 10))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.graphs_tab)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def initialize_tool(self):
        def init():
            try:
                self.llama_tool = LLaMA32TensorRTTool()
                self.master.after(0, self.load_or_initialize_conversation)
            except Exception as e:
                self.master.after(0, self.display_error, f"Error initializing tool: {str(e)}")

        threading.Thread(target=init, daemon=True).start()

    def load_or_initialize_conversation(self):
        if self.llama_tool.load_base_state():
            self.display_message("Previous conversation state loaded.")
            self.display_message("You can continue the conversation or start a new one using the 'Start New Conversation' button.")
            self.is_first_message = False
            self.update_status("Ready")
            self.update_time()
            self.update_emotion_label()
        else:
            self.display_message("No previous conversation found. Please provide a character description to start.")
            self.is_first_message = True
            self.update_status("Awaiting character description")

    def start_new_conversation(self):
        if messagebox.askyesno("New Conversation", "Are you sure you want to start a new conversation? This will erase the current state."):
            self.llama_tool = LLaMA32TensorRTTool()  # Reinitialize the tool
            self.is_first_message = True
            self.display_message("New conversation started. Please provide a character description.")
            self.update_status("Awaiting character description")

    def send_message(self, event=None):
        user_input = self.input_field.get()
        self.input_field.delete(0, tk.END)
        self.display_message(f"You: {user_input}")

        if user_input.lower() == 'exit':
            self.master.quit()
            return

        if self.llama_tool is None:
            self.display_message("Tool is not initialized yet. Please wait.")
            return

        if self.is_first_message:
            try:
                self.llama_tool.set_system_prompt(user_input)
                self.display_message("Character description set. You can now start interacting with the AI.")
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
            interaction_result = self.llama_tool.interact(user_input)
            
            self.master.after(0, self.display_message, f"AI: {interaction_result['response']}")
            self.master.after(0, self.update_emotion_label, interaction_result['emotion'])
            self.master.after(0, self.update_time, interaction_result['time'])
            self.master.after(0, self.check_sleep_status, interaction_result['sleep_info'])
            self.master.after(0, self.update_loss_plot, 
                              interaction_result['lm_loss'], 
                              interaction_result['refusal_loss'], 
                              interaction_result['validation_loss'],
                              interaction_result['is_refusal'],
                              interaction_result['iterations'])
            self.master.after(0, self.enable_feedback)
            
            if interaction_result['iterations'] > 1:
                self.master.after(0, self.display_message, f"(Response generated after {interaction_result['iterations']} attempts)")
        except Exception as e:
            error_message = f"Error generating response: {str(e)}\n{traceback.format_exc()}"
            self.master.after(0, self.display_message, error_message)
            logging.error(error_message)
        finally:
            self.master.after(0, lambda: self.send_button.config(state='normal'))
            self.master.after(0, self.update_status, "Ready")

    def update_loss_plot(self, lm_loss, refusal_loss, validation_loss, is_refusal, iterations):
        for ax in self.axes.flat:
            ax.clear()

        # Plot language modeling and validation losses
        self.axes[0, 0].plot(self.llama_tool.training_losses, label='LM Loss')
        self.axes[0, 0].plot(self.llama_tool.validation_losses, label='Validation Loss')
        self.axes[0, 0].legend()
        self.axes[0, 0].set_title('Language Modeling Loss')
        self.axes[0, 0].set_xlabel('Interactions')
        self.axes[0, 0].set_ylabel('Loss')

        # Plot refusal loss
        refusal_losses = [result['refusal_loss'] for result in self.llama_tool.interaction_results]
        self.axes[0, 1].plot(refusal_losses, label='Refusal Loss')
        self.axes[0, 1].legend()
        self.axes[0, 1].set_title('Refusal Loss')
        self.axes[0, 1].set_xlabel('Interactions')
        self.axes[0, 1].set_ylabel('Loss')

        # Plot overfitting indicator
        loss_diff = [v - t for v, t in zip(self.llama_tool.validation_losses, self.llama_tool.training_losses)]
        self.axes[1, 0].plot(loss_diff, label='Val Loss - LM Loss')
        self.axes[1, 0].axhline(y=0, color='r', linestyle='--')
        self.axes[1, 0].legend()
        self.axes[1, 0].set_title('Overfitting Indicator')
        self.axes[1, 0].set_xlabel('Interactions')
        self.axes[1, 0].set_ylabel('Loss Difference')

        # Plot refusal rate
        refusal_rate = [sum(self.llama_tool.refusal_history[max(0, i-99):i+1]) / min(100, i+1) for i in range(len(self.llama_tool.refusal_history))]
        self.axes[1, 1].plot(refusal_rate, label='Refusal Rate')
        self.axes[1, 1].set_ylim(0, 1)
        self.axes[1, 1].legend()
        self.axes[1, 1].set_title('Refusal Rate (100-interaction moving average)')
        self.axes[1, 1].set_xlabel('Interactions')
        self.axes[1, 1].set_ylabel('Refusal Rate')

        # Plot iterations per response
        iterations_history = [result['iterations'] for result in self.llama_tool.interaction_results]
        self.axes[2, 0].plot(iterations_history, label='Iterations')
        self.axes[2, 0].set_ylim(1, max(iterations_history) + 1)
        self.axes[2, 0].legend()
        self.axes[2, 0].set_title('Iterations per Response')
        self.axes[2, 0].set_xlabel('Interactions')
        self.axes[2, 0].set_ylabel('Iterations')

        self.fig.tight_layout()
        self.canvas.draw()

    def update_time(self, time):
        self.time_label.config(text=f"Current Time: {time:.2f}")

    def check_sleep_status(self, sleep_info):
        if sleep_info['should_sleep']:
            message = "It's night time. " if sleep_info['time_of_day'] >= 0.7 else ""
            message += "The model may be overfitting. " if sleep_info.get('overfitting', False) else ""
            message += "Consider letting the model sleep to consolidate learning."
            self.display_message(message)
            self.sleep_button.config(state='normal')
        else:
            self.sleep_button.config(state='disabled')

    def sleep_kan(self):
        if self.llama_tool:
            message = self.llama_tool.perform_sleep()
            self.display_message(message)
            self.update_time()
            self.sleep_button.config(state='disabled')

    def save_kan_state(self):
        if self.llama_tool:
            self.llama_tool.save_kan_state()
            self.display_message("KAN state saved.")

    def load_saved_state(self):
        if self.llama_tool:
            filename = filedialog.askopenfilename(
                initialdir=self.llama_tool.kan_state_dir,
                title="Select KAN State to Load",
                filetypes=[("PyTorch State", "*.pt")]
            )
            if filename:
                if self.llama_tool.load_base_state():
                    self.display_message(f"KAN state loaded: {filename}")
                    self.update_time()
                    self.update_emotion_label()
                    self.is_first_message = False
                else:
                    self.display_message("Failed to load KAN state. Please try again.")

    def enable_feedback(self):
        self.feedback_button.config(state='normal')

    def submit_feedback(self):
        if self.llama_tool:
            try:
                emotion_feedback = [float(x) for x in self.emotion_feedback_entry.get().split(',')]
                compliance = float(self.compliance_entry.get())
                
                if len(emotion_feedback) == 2 and 0 <= compliance <= 1:
                    self.llama_tool.update_emotional_state(emotion_feedback)
                    self.display_message("Feedback submitted successfully.")
                    self.emotion_feedback_entry.delete(0, tk.END)
                    self.compliance_entry.delete(0, tk.END)
                else:
                    self.display_message("Invalid feedback format. Please try again.")
            except ValueError:
                self.display_message("Invalid input. Please enter valid numbers.")
        else:
            self.display_message("Tool not initialized. Please wait.")

    def update_emotion_label(self, emotion=None):
        if emotion is None and self.llama_tool:
            emotion = self.llama_tool.get_current_emotion()
        self.emotion_label.config(text=f"Emotion: {emotion}")

    def update_status(self, status):
        self.status_label.config(text=f"Status: {status}")

    def display_message(self, message):
        self.chat_display.configure(state='normal')
        self.chat_display.insert(tk.END, message + "\n")
        self.chat_display.configure(state='disabled')
        self.chat_display.see(tk.END)
        logging.info(message)

    def display_error(self, message):
        self.display_message(message)
        self.update_status("Error")
        messagebox.showerror("Error", message)
        logging.error(message)
        logging.error(traceback.format_exc())

def main():
    root = tk.Tk()
    gui = LLAMA32GUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()