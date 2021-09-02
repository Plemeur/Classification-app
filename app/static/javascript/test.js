const transformSelector = document.getElementById('transformSelector');
const submitButton = document.getElementById('submitButton');
const inputs = document.getElementsByTagName('input');
const lossSelector = document.getElementById('lossSelector');
const optimizerSelector = document.getElementById('optimizerSelector');
const schedulerSelector = document.getElementById('schedulerSelector');

// Rewrite this to match the style of the other one
function displayTransformParameters(e) {
    const transformParameters = document.createElement('div');
    const count = parseInt(e.target.name.substring(17));
    const transform = e.target.value;
    const transformSection = e.target.parentElement;

    transformParameters.classList.add('parameters');
    transformParameters.id = `transformParameters${count}`    
    
    fetch(`parameters?type=transform&choice_name=${transform}`)
    .then(response => {
        return response.text();
    })
    .then(string => {
        let newSelector = transformSelector.cloneNode(true);
        const newName = `transformSelector${count+1}`;

        if (document.querySelector(`[name  = ${newName}]`) == null) {
            newSelector.addEventListener('change', displayTransformParameters);
            newSelector.name = newName;
            transformParameters.innerHTML = string;
            transformSection.appendChild(transformParameters);
            transformSection.appendChild(newSelector);
        } else {
            currentParameters = document.getElementById(`transformParameters${count}`);
            console.log(currentParameters);
            currentParameters.innerHTML = string;
        }        
    })    
}

function displayParameters(e) {
    const param = e.target.name.substring(0,e.target.name.length-5);
    let parameters = document.querySelector(`#${param}Parameters`);
    
    if (!parameters) {
        const paramSection = e.target.parentElement;
        parameters = document.createElement('div');
        parameters.classList.add('parameters');
        parameters.id = `${param}Parameters`;
        paramSection.appendChild(parameters);
    }

    const transform = e.target.value;
    
    fetch(`parameters?type=${param}&choice_name=${transform}`)
    .then(response => {
        return response.text();
    })
    .then(string => {
        parameters.innerHTML = string;        
    })
}

function postForm(e) {
    e.preventDefault()
    const data = new FormData(e.target.parentElement);
    const values = Object.fromEntries(data.entries());
    
    fetch( '/training', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(values)
    })
}

function makeBoolean(e) {
    const box = e.srcElement;
    if (box.checked){
        box.value=true;
    } else {
        box.value=false;
    }
}



// Display Parameter on selection
transformSelector.addEventListener('change', displayTransformParameters);
lossSelector.addEventListener('change', displayParameters);
optimizerSelector.addEventListener('change', displayParameters);
schedulerSelector.addEventListener('change', displayParameters);

// send form without reload
submitButton.addEventListener('click', postForm);

// Apply boolean to checkboxes
for(let i = 0; i < inputs.length; i++) {
    if(inputs[i].type == "checkbox") {
        inputs[i].addEventListener('change', makeBoolean); 
    }  
}