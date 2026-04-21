const chatArea = document.getElementById('chat-area');
const dynamicForm = document.getElementById('dynamic-form');
const submitBtn = document.getElementById('submit-form');
const inputArea = document.getElementById('input-area');
const chatWidget = document.getElementById('chat-widget');
let currentFlow = null;

function toggleChat() {
    chatWidget.classList.toggle('active');
}

function addMessage(text, isUser = false) {
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${isUser ? 'user-message' : 'ai-message'} fade-in-up`;
    const bubble = document.createElement('div');
    bubble.className = 'msg-bubble';
    bubble.innerHTML = text; // allow html
    msgDiv.appendChild(bubble);
    chatArea.insertBefore(msgDiv, chatArea.lastElementChild);
    chatArea.appendChild(document.querySelector('.options-grid'));
    chatArea.scrollTop = chatArea.scrollHeight;
}

function startFlow(flow) {
    currentFlow = flow;
    inputArea.style.display = 'block';
    submitBtn.style.display = 'block';
    dynamicForm.innerHTML = ''; // reset
    
    if(flow === 'upload') {
        addMessage("Please upload your Medical Report (PDF/Image).", false);
        dynamicForm.innerHTML = `<div class="form-group"><input type="file" id="report_file" accept=".pdf,image/*"></div>`;
    } 
    else if(flow === 'diagnose') {
        addMessage("Sure. Please provide the following clinical details to assess your cardiovascular risk:", false);
        dynamicForm.innerHTML = `
            <div class="form-group"><label>Age</label><input type="number" id="age" value="45"></div>
            <div class="form-group"><label>Gender (0=F, 1=M)</label><input type="number" id="gender" value="1"></div>
            <div class="form-group"><label>Height (cm)</label><input type="number" id="height" value="175"></div>
            <div class="form-group"><label>Weight (kg)</label><input type="number" id="weight" value="80"></div>
            <div class="form-group"><label>Systolic BP</label><input type="number" id="sys" value="130"></div>
            <div class="form-group"><label>Diastolic BP</label><input type="number" id="dia" value="85"></div>
            <div class="form-group"><label>Blood Sugar</label><input type="number" id="sugar" value="110"></div>
            <div class="form-group"><label>Heart Rate</label><input type="number" id="hr" value="75"></div>
            <div class="form-group"><label>Stress Level (1-10)</label><input type="number" id="stress" value="5"></div>
        `;
    }
    else if(flow === 'quickcheck') {
        addMessage("Quick BP and Sugar Check. Please enter your vital signs below:", false);
        dynamicForm.innerHTML = `
            <div class="form-group"><label>Systolic BP</label><input type="number" id="q_sys" placeholder="e.g. 120"></div>
            <div class="form-group"><label>Diastolic BP</label><input type="number" id="q_dia" placeholder="e.g. 80"></div>
            <div class="form-group"><label>Blood Sugar Level</label><input type="number" id="q_sugar" placeholder="e.g. 95"></div>
        `;
    }
    else if(flow === 'medicine') {
        addMessage("What medicine would you like information about?", false);
        dynamicForm.innerHTML = `
            <div class="form-group"><label>Medicine Name</label><input type="text" id="med_name" placeholder="e.g. Aspirin"></div>
        `;
    }
}

submitBtn.addEventListener('click', async () => {
    addMessage("Processing your request...", true);
    inputArea.style.display = 'none';
    
    if(currentFlow === 'diagnose') {
        const payload = {
            Age: +document.getElementById('age').value,
            Gender: +document.getElementById('gender').value,
            gender2: +document.getElementById('gender').value,
            Height_cm: +document.getElementById('height').value,
            Weight_kg: +document.getElementById('weight').value,
            occupation_encoded: 0, Hypertension: 0,
            systolic_bp: +document.getElementById('sys').value,
            diastolic_bp: +document.getElementById('dia').value,
            total_cholesterol: 200,
            blood_sugar: +document.getElementById('sugar').value,
            heart_rate: +document.getElementById('hr').value,
            diabetic_encoded: 0, smoking_encoded: 0,
            If_smoker_cigarettes_per_day: 0, alcohol_encoded: 0,
            physical_activity: 1, Average_Sleep_Duration: 7,
            Stress_Level_Self_Assessment: +document.getElementById('stress').value,
            diet_type: 1, Steps_in_a_day: 5000,
            family_history: 0, Medical_Condition: -1
        };
        
        try {
            let res = await fetch('http://127.0.0.1:8001/predict', {
                method: 'POST',
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify(payload)
            });
            let data = await res.json();
            
            let explanationsHtml = '';
            if (data.prediction === 1 && data.explanation && data.explanation.length > 0) {
                explanationsHtml = `<p><strong>High Risk Factors:</strong></p>
                <ul>${data.explanation.map(factor => `<li>${factor}</li>`).join('')}</ul>`;
            }

            let html = `<div class="card-result">
                <h3>Risk Level: ${data.risk_percentage.toFixed(1)}%</h3>
                <p><strong>Diagnosis:</strong> ${data.prediction === 1 ? 'Elevated Risk' : 'Normal Risk'}</p>
                ${explanationsHtml}
                <p><strong>Recommendations:</strong></p>
                <ul>${data.recommendations.map(r => `<li>${r}</li>`).join('')}</ul>
                <small>${data.disclaimer}</small>
            </div>`;
            addMessage(html, false);
        } catch(e) { addMessage("Error reaching server.", false); }
    }
    else if(currentFlow === 'quickcheck') {
        const payload = {
            systolic_bp: +document.getElementById('q_sys').value,
            diastolic_bp: +document.getElementById('q_dia').value,
            blood_sugar: +document.getElementById('q_sugar').value
        };
        try {
            let res = await fetch('http://127.0.0.1:8001/quick-check', { method: 'POST', headers: {"Content-Type": "application/json"}, body: JSON.stringify(payload) });
            let data = await res.json();
            
            let html = `<div class="card-result">
                <h3>Status: ${data.risk_level} Risk</h3>
                <p>BP Status: ${data.bp_status} | Sugar: ${data.sugar_status}</p>
                <p><strong>Recommendations:</strong></p>
                <ul>${data.recommendations.map(r => `<li>${r}</li>`).join('')}</ul>
                ${data.alert ? `<p style="color:red; font-weight:bold">${data.alert}</p>` : ''}
            </div>`;
            addMessage(html, false);
        } catch(e) { addMessage("Error reaching server.", false); }
    }
    else if(currentFlow === 'medicine') {
        const payload = { medicine_name: document.getElementById('med_name').value };
        try {
            let res = await fetch('http://127.0.0.1:8001/medicine-info', { method: 'POST', headers: {"Content-Type": "application/json"}, body: JSON.stringify(payload) });
            let data = await res.json();
            
            let html = `<div class="card-result">
                <h3>${data.medicine_name}</h3>
                <p><strong>Why used:</strong> ${data.why_it_is_used}</p>
                <p><strong>Guidance:</strong> ${data.dosage_info}</p>
                <p><strong>Side Effects:</strong> ${data.side_effects}</p>
                <small style="color:#ec4899;">${data.warning}</small>
            </div>`;
            addMessage(html, false);
        } catch(e) { addMessage("Error reaching server.", false); }
    }
    else if(currentFlow === 'upload') {
        setTimeout(() => {
            addMessage("OCR Analysis simulated. Found generic metrics.", false);
        }, 1000);
    }
});
