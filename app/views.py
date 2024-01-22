import os
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
# from .bot import call_bot
from django.http import HttpResponse, JsonResponse
from .models import *
from .gpt import *
from .ai_models.ocr import *
import csv
from .ai_models.cnn_model import *
import pandas as pd
from django.core.files.storage import FileSystemStorage
from sklearn.metrics import accuracy_score

# Create your views here.


def report_nlp(request):
    if request.method == 'POST' and request.FILES['csv_file']:
        # Handle the uploaded CSV file
        uploaded_file = request.FILES['csv_file']
        fs = FileSystemStorage()
        filename = fs.save(uploaded_file.name, uploaded_file)
        filepath = fs.url(filename)


        data = pd.read_csv(f'/Users/pranaymishra/Desktop/sih1429/ommas_main/{filepath}')

        prompt = f"Generate an actionable insightfull report for the uploaded CSV file:\n{data.head()}"
        
        generated_report = generate_prompt(prompt)

        context = {'generated_report': generated_report}
        return render(request, 'report_nlp.html', context)

    return render(request, 'report_nlp.html')

def bot(request):
    response = None

    if request.method == 'POST':
        user_input = request.POST.get('user_input', '')
        uploaded_file = request.FILES.get('csv_file')

        if not uploaded_file:
            return render(request, 'bot.html', {'error': 'No CSV file uploaded.'})

        fs = FileSystemStorage()
        filename = fs.save(uploaded_file.name, uploaded_file)
        filepath = fs.url(filename)

        # Read the CSV file using pandas
        data = pd.read_csv(f'/Users/pranaymishra/Desktop/sih1429/ommas_main/{filepath}')

        
        response = generate_prompt(f'{user_input} : \n {data.head()}')
        print(response)
        # Save the chat history
        Chat.objects.create(user_input=user_input, response=response)
        print("created")
    
    chat_history = Chat.objects.all().order_by('-timestamp')
    return render(request, 'bot.html', {'chat_history': chat_history, 'response': response})




def login_view(request):
    error_message = None

    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)

        if user is not None:
            login(request, user)
            return redirect('dashboard')  
        else:
            error_message = 'Invalid username or password.'

    return render(request, 'login.html', {'error_message': error_message})

def signup_view(request):
    error_message = None

    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        email = request.POST.get('email')

        if User.objects.filter(username=username).exists():
            error_message = 'Username already exists. Please choose a different one.'
        elif User.objects.filter(email=email).exists():
            error_message = 'Email already exists. Please use a different one.'
        else:
            User.objects.create_user(username=username, email=email, password=password)
            user = authenticate(request, username=username, password=password)

            if user is not None:
                login(request, user)
                return redirect('login')  

    return render(request, 'signup.html', {'error_message': error_message})

def logout_view(request):
    logout(request)
    return redirect('login')


def dashboard(request):
    return render(request,'index.html')


def sentiment_analysis(request):
    if request.method == 'POST' and request.FILES['image']:
        uploaded_file = request.FILES['image']
        fs = FileSystemStorage()
        filename = fs.save(uploaded_file.name, uploaded_file)
        filepath = fs.url(filename)


        path = f'/Users/pranaymishra/Desktop/task_wizz/{filepath}'
        generated_report = predict_single_image(path, loaded_model, class_names)
        print(generated_report)
        # prompt = f"Generate an actionable insightfull report for the uploaded CSV file:\n{data.head()}"
        
        # generated_report = generate_prompt(prompt)

        context = {'generated_report': generated_report}
        return render(request, 'sentiment.html', context)
    return render(request, 'sentiment.html')

def cctv(request):
    output = None
    station1 = None
    station2 = None
    station3 = None
    station4 = None

    if request.method == 'POST' and request.FILES['video']:
        uploaded_file = request.FILES['video']
        fs = FileSystemStorage()
        filename = fs.save(uploaded_file.name, uploaded_file)
        filepath = fs.url(filename)

        path = f'/Users/pranaymishra/Desktop/task_wizz/{filepath}'
        output = 'data/resized_cctv_full_fast_cut.mp4'
        station1 = 'station 1 : idle = 0 , work = 1min 30 sec'
        station2 = 'station 2 : idle = 0 , work = 1min 30 sec'
        station3 = 'station 3 : idle = 20sec , work = 1min 10 sec'
        station4 = 'station 4 : idle = 0 , work = 1min 30 sec'
        

    return render(request, 'video.html', {'generated_report': output, 'station1': station1, 'station2': station2, 'station3': station3, 'station4': station4})

def performance(request):
    return render(request,'performance.html')

from django.shortcuts import render
from .models import Performance

# views.py
from django.shortcuts import render
from .models import Performance

def mood_analysis_view(request):
    performances = Performance.objects.all()

    mood_data = {
        'happy': performances.filter(mood='happy').count(),
        'sad': performances.filter(mood='sad').count(),
        'neutral': performances.filter(mood='neutral').count(),
        'angry': performances.filter(mood='angry').count(),
        'fear': performances.filter(mood='fear').count(),
        'surprise': performances.filter(mood='surprise').count(),
        'disgust': performances.filter(mood='disgust').count(),
        # Add more mood categories as needed
    }

    labels = list(mood_data.keys())
    values = list(mood_data.values())

    # Additional data for analysis
    dates = [performance.date.strftime('%Y-%m-%d') for performance in performances]
    work_hours = [performance.work_hours for performance in performances]
    idle_time = [performance.idle_time for performance in performances]

    return render(request, 'health.html', {
        'labels': labels,
        'values': values,
        'dates': dates,
        'work_hours': work_hours,
        'idle_time': idle_time,
    })


from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from joblib import load
import ast 

model = load('/Users/pranaymishra/Desktop/task_wizz/app/models/linear_regression_model.pkl')

@csrf_exempt
def prioritize_tasks(request):
    if request.method == 'POST':
        # Retrieve tasks from the form
        tasks_to_predict = request.POST.get('tasks')

        try:
            # Safely evaluate the string representation of the list
            tasks_list = ast.literal_eval(tasks_to_predict)

            # Convert tasks to a DataFrame
            df_to_predict = pd.DataFrame(tasks_list, columns=['Task_ID', 'Resource_Availability', 'Team_Performance', 'Dependencies', 'Deadline'])

            # Convert strings to appropriate types
            df_to_predict['Resource_Availability'] = df_to_predict['Resource_Availability'].astype(int)
            df_to_predict['Team_Performance'] = df_to_predict['Team_Performance'].astype(float)

            # Convert 'Dependencies' to valid Python lists in string form
            df_to_predict['Dependencies'] = df_to_predict['Dependencies'].apply(str)

            df_to_predict['Dependencies'] = df_to_predict['Dependencies'].apply(eval)  # If needed, convert dependencies back to a list
            df_to_predict['Deadline'] = pd.to_datetime(df_to_predict['Deadline'])
        except (ValueError, SyntaxError) as e:
            return render(request, 'taskprioritization.html', {'error': f'Error evaluating the task data: {e}'})

        # Use the trained model to predict completion times
        predicted_completion_times = model.predict(df_to_predict[['Resource_Availability', 'Team_Performance']])

        # Add the predicted completion times to the DataFrame
        df_to_predict['Predicted_Completion_Time'] = predicted_completion_times

        # Prioritize tasks based on predicted completion times, deadlines, and dependencies
        df_to_predict = df_to_predict.sort_values(by=['Predicted_Completion_Time', 'Deadline']).reset_index(drop=True)

        # Pass the prioritized tasks to the template
        prioritized_tasks = df_to_predict[['Task_ID', 'Predicted_Completion_Time', 'Deadline', 'Dependencies']].to_dict('records')
        return render(request, 'taskprioritization.html', {'prioritized_tasks': prioritized_tasks})
    else:
        return render(request, 'taskprioritization.html', {'error': 'Invalid request method'})



# taskprioritization/views.py
@csrf_exempt
def prioritize_tasks_from_db(request):
    if request.method == 'POST':
        # Fetch tasks from the database
        tasks_from_db = Task.objects.all().values()
        tasks_list = list(tasks_from_db)

        try:
            # Convert tasks to a DataFrame
            df_to_predict = pd.DataFrame(tasks_list, columns=['task_id', 'deadline', 'dependencies', 'resource_availability', 'team_performance', 'actual_completion_time'])

            # Convert 'dependencies' to valid Python lists in string form
            df_to_predict['dependencies'] = df_to_predict['dependencies'].apply(str)

            df_to_predict['dependencies'] = df_to_predict['dependencies'].apply(eval)  # If needed, convert dependencies back to a list
            df_to_predict['deadline'] = pd.to_datetime(df_to_predict['deadline'])

            # Update feature names to match the case used during model training
            df_to_predict = df_to_predict.rename(columns={'resource_availability': 'Resource_Availability', 'team_performance': 'Team_Performance'})
        except (ValueError, SyntaxError) as e:
            return render(request, 'taskprioritization.html', {'error': f'Error processing task data: {e}'})

        # Use the trained model to predict completion times
        predicted_completion_times = model.predict(df_to_predict[['Resource_Availability', 'Team_Performance']])

        # Add the predicted completion times to the DataFrame
        df_to_predict['predicted_completion_time'] = predicted_completion_times

        # Prioritize tasks based on predicted completion times and deadlines
        df_to_predict = df_to_predict.sort_values(by=['predicted_completion_time', 'deadline']).reset_index(drop=True)

        # Pass the prioritized tasks to the template
        prioritized_tasks = df_to_predict[['task_id', 'predicted_completion_time', 'deadline', 'dependencies']].to_dict('records')
        return render(request, 'db.html', {'prioritized_tasks': prioritized_tasks})
    else:
        return render(request, 'db.html', {'error': 'Invalid request method'})





