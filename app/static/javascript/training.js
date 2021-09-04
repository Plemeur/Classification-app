const transformSelector = document.getElementById('transformSelector');
const valTransformSelector = document.getElementById('valTransformSelector');
const submitButton = document.getElementById('submitButton');
const inputs = document.getElementsByTagName('input');
const lossSelector = document.getElementById('lossSelector');
const optimizerSelector = document.getElementById('optimizerSelector');
const schedulerSelector = document.getElementById('schedulerSelector');

// Rewrite this to match the style of the other one
function displayTransformParameters(e) {
    const transformSection = e.target.parentElement.parentElement;
    const transformSubSection = e.target.parentElement;
    const count = parseInt(e.target.name.match(/\d+/));
    const newName = e.target.name.replace(count, count+1);
    const parametersId = e.target.name.replace(count, `Parameters${count}`)
    console.log(e.target.name)

    const transformParameters = document.createElement('div');
    transformParameters.classList.add('parameters');
    transformParameters.id = parametersId

    const transform = e.target.value;    
    fetch(`parameters?type=transform&choice_name=${transform}&section=${e.target.name}`)
    .then(response => {
        return response.text();
    })
    .then(string => {
        if (document.querySelector(`[name  = ${newName}]`) == null) {
            let newSubSection = transformSubSection.cloneNode(false);
            let newSelector = transformSelector.cloneNode(true);

            newSelector.addEventListener('change', displayTransformParameters);
            newSelector.name = newName;

            transformParameters.innerHTML = string;
            transformSubSection.appendChild(transformParameters);
            
            newSubSection.appendChild(newSelector);
            transformSection.appendChild(newSubSection);
            console.log(transformParameters.childNodes);
        } else {
            currentParameters = document.getElementById(parametersId);
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
valTransformSelector.addEventListener('change', displayTransformParameters);
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