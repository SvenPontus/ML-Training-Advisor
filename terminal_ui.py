from ml_class_pipeline import *
from make_pandas import MakePandas
from validation_ml_program import Validation as Vald
import sys

r_or_c_list = list()

# Function to validate if user input is 'r' or 'c'
def validate_r_or_c(user_input):
    try:
        return Vald.validate_user_input_r_or_c(user_input)
    except ValueError as e:
        print(e)

# code for frontend in terminal
def get_user_r_or_c_frontend():
    # User-friendly with 5 tries
    max_attempts = 5
    attempts = 0
    
    while attempts < max_attempts:
        print("\nFind the best regressor or best classifier model for your data!")
        print("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
        print("Is it a regressor (r) or classifier (c) model you need for your data? ")
        
        user_input = input("Please choose 'r' for regressor or 'c' for classifier: ")
        
        try:
            # Validate user input and append it if valid
            validated_input = validate_r_or_c(user_input)
            if validated_input:
                r_or_c_list.append(validated_input)
                return  # Exit function once a valid input is provided
        except ValueError as e:
            print(f"Error: {e}. Please try again.")
        
        # Increment attempts after handling the invalid input
        attempts += 1
    
    # If the loop exits due to max_attempts being reached, terminate the program
    print("Too many invalid attempts. The program will now terminate.")
    sys.exit(1)
        


def run_app():
    while True:

        # User input if regressor or classifier
        get_user_r_or_c_frontend()
        # Get csv file and check if .csv
        try:
            csv_name = input("Upload your csv file, dont forget .csv : ")
            csv_file = MakePandas(Vald.controll_csv(csv_name))
            df = csv_file.read_csv_pandas()
        except Exception as e:
            print(e)
        print("Which number is your Target Label?")
        # Print out features
        for nr,_ in enumerate(df.columns):
            print(f"{nr} {_}")
        target_label = Vald.read_in_int_value(Vald.validate_int, "Enter your choice : ")

        control_reg_or_cat = MakePandas.controll_reg_or_cat(target_value=target_label,r_or_c=r_or_c_list[-1])
        print(control_reg_or_cat)
        # Quit program if user load wrong csv for the operation
        if "ERROR!" in control_reg_or_cat[:6]:
            print("You have load wrong csv for this operation")
            input("Enter to Quit the program")
            break 
        input("press enter")
        
        dummies_nan_values_or_ready, dummies = MakePandas.ready_for_ml_and_dummies(target_label)
        for _ in dummies_nan_values_or_ready:
            print(_)
        # print out and give user opportunity to do dummies on the file if needed, and if yes do the whole process
        if dummies:
            y_or_no = Vald.read_in_str_value(Vald.validate_yes_or_no, "Do you want do to dummies? ")
            if y_or_no == "yes":
                df = MakePandas(csv_name)
                df.read_csv_pandas()
                MakePandas.do_dummies(target_label)
                ml_output = MakePandas.pick_up_target_split_and_call_ml(target_label,r_or_c_list[-1])
                print(ml_output)
                input("")
                break
            elif y_or_no == "no":
                continue
        # If file not ready for ml, Quit program    
        if not "Your file is ready for ML." in dummies_nan_values_or_ready:
            print("Fix your csv file")
            input("Enter to Quit the program")
            break 
        input("press enter and wait for the program to create report for cost functions on models")
        
        ml_output = MakePandas.pick_up_target_split_and_call_ml(target_label,r_or_c_list[-1])
        # Print out cost functions report
        print(ml_output)
        
        input("Read your cost functions and then, press enter")
        r_best_model = MakePandas.BEST_MODEL[-1]
        print(r_best_model)
        dump_best_model_user = Vald.read_in_str_value(Vald.validate_yes_or_no, "Do you want to dump best modell? ")
        # Opportunity to create a file of the best model
        if dump_best_model_user.lower() == "yes":
            MakePandas.dump_best_model_final()
            print("Now you modell is dump")
        elif dump_best_model_user.lower()  == "no":
            input("press enter to quit the program")        
        break
        
if __name__ == "__main__":
    run_app()



