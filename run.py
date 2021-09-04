from flask import Flask, render_template, request, redirect, jsonify, \
    url_for, flash, render_template_string

import random
import string
import logging
import json
import httplib2
import requests

from src.losses.get_loss import all_losses_dict
from src.models.get_model import all_models_dict
from src.optimizers.get_optimizer import all_optim_dict
from src.transforms.get_transforms import all_transforms_dict
from src.schedulers.get_scheduler import all_scheduler_dict
from app.utils.parameters import parse_parameters
from train import train

SUCCESS_RESPONSE = json.dumps({'success':True}), 200, {'ContentType':'application/json'} 

app = Flask(__name__, template_folder='app/templates', static_folder='app/static')
app.jinja_env.globals.update(zip=zip) # Allows Jinja to uses the python zip function

# Display all things
@app.route('/')
def home():
    return render_template('index.html',
        losses = all_losses_dict,
        models = all_models_dict,
        transforms = all_transforms_dict,
        optimizers = all_optim_dict,
        schedulers= all_scheduler_dict )


@app.route('/parameters')
def get_transforms():
    section = request.args['type'] 
    if section == 'transform':  
        func = all_transforms_dict[request.args['choice_name']]
        section = request.args['section']
    elif section == 'optimizer':
        func = all_optim_dict[request.args['choice_name']]
    elif section == 'loss':
        func = all_losses_dict[request.args['choice_name']]
    elif section == 'scheduler':
        func = all_scheduler_dict[request.args['choice_name']]

    try : 
        params = func.__init__.__code__.co_varnames
        params = params[:func.__init__.__code__.co_argcount]
        defaults = list(func.__init__.__defaults__)
        # Default arguments are always at the end of the function
        while len(defaults)<len(params):
            defaults.insert(0, None)
    except :
        params = []
        defaults = [] 

    # Looks like HTML does not like spaces
    defaults = list(map(lambda x: f'{x}'.replace(' ',''), defaults))
    print(params)
    return render_template('parameters.html',
        params = params, 
        defaults=defaults, 
        section=section)

@app.route('/training', methods=['post'])
def training():
    parameters = parse_parameters(request.data)
    print(parameters)
    train(parameters)
    
    return SUCCESS_RESPONSE

if __name__ == '__main__':
    app.secret_key = 'super_secret_key'
    app.debug = True
    app.run(host='0.0.0.0', port=8000)