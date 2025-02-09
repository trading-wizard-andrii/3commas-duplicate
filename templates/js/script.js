// Expand/Collapse Sections
document.querySelectorAll('.section h2').forEach(header => {
    header.addEventListener('click', () => {
        const content = header.nextElementSibling;
        const arrow = header.querySelector('.toggle-arrow');

        content.style.display = content.style.display === 'block' ? 'none' : 'block';
        arrow.classList.toggle('expanded');
    });
});

function toggleElement(trigger, target, options = {}) {
    const { activeClass = "active", showStyle = "block", hideStyle = "none" } = options;

    trigger.addEventListener("click", () => {
        const isVisible = target.style.display === showStyle;
        target.style.display = isVisible ? hideStyle : showStyle;

        if (activeClass) {
            trigger.classList.toggle(activeClass, !isVisible);
        }
    });
}

function toggleExclusiveButtons(btn1, btn2, config1, config2, options = {}) {
    const { activeClass = "active", showStyle = "block", hideStyle = "none" } = options;

    btn1.addEventListener("click", () => {
        config1.style.display = showStyle;
        config2.style.display = hideStyle;

        btn1.classList.add(activeClass);
        btn2.classList.remove(activeClass);
    });

    btn2.addEventListener("click", () => {
        config2.style.display = showStyle;
        config1.style.display = hideStyle;

        btn2.classList.add(activeClass);
        btn1.classList.remove(activeClass);
    });
}

document.addEventListener("DOMContentLoaded", () => {
    const priceChangeBtn = document.getElementById("price-change-btn");
    const conditionsBtn = document.getElementById("conditions-btn");
    const priceChangeConfig = document.getElementById("price-change-config");
    const conditionsConfig = document.getElementById("conditions-config");
    toggleExclusiveButtons(
        priceChangeBtn,
        conditionsBtn,
        priceChangeConfig,
        conditionsConfig,
        { activeClass: "active" }
    );
    // Dropdown functionality
    const dropdownToggle = document.getElementById("dropdown-toggle");
    const dropdownList = document.getElementById("dropdown-list");
    toggleElement(dropdownToggle, dropdownList, { activeClass: "active", showStyle: "block", hideStyle: "none" });

    // Close dropdown if clicked outside
    document.addEventListener("click", (event) => {
        if (!dropdownToggle.contains(event.target) && !dropdownList.contains(event.target)) {
            dropdownList.classList.remove("active");
        }
    });

    // Populate dropdown with pairs
    fetch("/get_crypto_pairs")
        .then((response) => {
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            return response.json();
        })
        .then((pairs) => {
            if (!Array.isArray(pairs) || pairs.length === 0) {
                dropdownList.innerHTML = "<div class='dropdown-item'>No pairs available</div>";
                return;
            }

            // After .map(...) creation of each .dropdown-item with a checkbox:
            dropdownList.innerHTML = pairs
                .map(pair => `
                    <div class="dropdown-item">
                        <input type="checkbox" value="${pair}" style="display:none;">
                        <span>${pair}</span>
                        <span class="checkmark" style="display: none;">✔</span>
                    </div>`
                )
                .join("");

            // Then in the items.forEach(...) block:
            const items = dropdownList.querySelectorAll(".dropdown-item");
            items.forEach(item => {
                item.addEventListener("click", () => {
                    // 1) Toggle the hidden checkbox
                    const chk = item.querySelector('input[type="checkbox"]');
                    if (chk) {
                        chk.checked = !chk.checked;
                    }

                    // 2) Toggle the .selected class (purely visual)
                    item.classList.toggle("selected");

                    // 3) Toggle the checkmark icon
                    const checkmark = item.querySelector(".checkmark");
                    checkmark.style.display = (checkmark.style.display === "none") ? "inline" : "none";
                });
            });
        })
});

// Reusable Function to Add Conditions Dynamically
function addCondition(buttonId, targetGroupId) {
    const button = document.getElementById(buttonId);
    const targetGroup = document.getElementById(targetGroupId);

    button.addEventListener('click', () => {
        // Create a new condition container
        const conditionContainer = document.createElement('div');
        conditionContainer.className = 'condition';

        // Add indicator selector
        const indicatorSelect = document.createElement('select');
        indicatorSelect.className = 'indicator-select';
        indicatorSelect.innerHTML = `
            <option value="RSI">RSI (Relative Strength Index)</option>
            <option value="TradingView">TradingView Crypto Screener</option>
            <option value="HeikenAshi">Heiken Ashi Candlesticks</option>
            <option value="MA">Moving Average (MA)</option>
            <option value="BollingerBands">Bollinger Bands %B</option>
            <option value="MACD">MACD (Moving Average Convergence Divergence)</option>
            <option value="Stochastic">Stochastic Oscillator</option>
            <option value="ParabolicSAR">Parabolic SAR (Stop and Reverse)</option>
        `;
        conditionContainer.appendChild(indicatorSelect);

        // Add dynamic inputs based on the selected indicator
        const dynamicInputs = document.createElement('div');
        dynamicInputs.className = 'dynamic-inputs';
        conditionContainer.appendChild(dynamicInputs);

        // Update inputs based on the selected indicator
        indicatorSelect.addEventListener('change', () => {
            const selectedIndicator = indicatorSelect.value;
            dynamicInputs.innerHTML = indicatorConfigs[selectedIndicator] || '';
        });

        // Trigger initial input update
        indicatorSelect.dispatchEvent(new Event('change'));

        // Add a delete button
        const deleteButton = document.createElement('button');
        deleteButton.className = 'delete-condition-btn';
        deleteButton.textContent = 'Delete';
        deleteButton.addEventListener('click', () => {
            targetGroup.removeChild(conditionContainer);
        });
        conditionContainer.appendChild(deleteButton);

        // Append the condition container to the condition group
        targetGroup.appendChild(conditionContainer);
    });
}

// Add Conditions Using the Reusable Function
addCondition('add-entry-condition', 'entry-condition-group');
addCondition('add-safety-condition', 'safety-condition-group');
addCondition('add-exit-condition', 'exit-condition-group');


// Enable/Disable Trailing Deviation Input
document.getElementById('trailing-toggle').addEventListener('change', function () {
    const trailingDeviation = document.getElementById('trailing-deviation');
    trailingDeviation.disabled = !this.checked;
});

// Function to generate indicator configurations
function createConfig(options) {
    return `
        <div class="input-group">
            ${options
                .map(option => {
                    if (option.type === "select") {
                        return `
                            <label>${option.label}</label>
                            <select id="${option.id || ""}">
                                ${option.values
                                    .map(value => `<option value="${value}">${value}</option>`)
                                    .join("")}
                            </select>
                        `;
                    } else if (option.type === "number") {
                        return `
                            <label>${option.label}</label>
                            <input type="number" id="${option.id || ""}" placeholder="${option.placeholder || ""}">
                        `;
                    }
                    return "";
                })
                .join("")}
        </div>
    `;
}

// Indicator Configurations
// Indicator Configurations
const indicatorConfigs = {
    RSI: createConfig([
        {
            type: "select",
            label: "RSI Length",
            values: [7, 14, 21, 28]
        },
        {
            type: "select",
            label: "Timeframe",
            values: ["1m", "5m", "15m", "1h", "4h", "1d"]
        },
        {
            type: "select",
            label: "Condition",
            values: ["Less Than", "Greater Than", "Crossing Down", "Crossing Up"]
        },
        {
            type: "number",
            label: "Signal Value",
            placeholder: "Enter Signal Value"
        }
    ]),

    TradingView: createConfig([
        {
            type: "select",
            label: "Timeframe",
            values: ["1m", "5m", "15m", "1h", "4h", "1d"]
        },
        {
            type: "select",
            label: "Signal Value",
            values: ["Buy", "Strong Buy", "Sell", "Strong Sell"]
        }
    ]),

    HeikenAshi: createConfig([
        {
            type: "select",
            label: "Candles in a Row",
            values: Array.from({ length: 10 }, (_, i) => i + 1)
        },
        {
            type: "select",
            label: "Timeframe",
            values: ["1m", "5m", "15m", "1h", "4h", "1d"]
        }
    ]),

    MA: createConfig([
        {
            type: "select",
            label: "MA Type",
            values: ["EMA", "SMA"]
        },
        {
            type: "select",
            label: "Fast MA",
            values: [5, 10, 14, 20, 25, 30, 50, 75, 100, 150, 200, 250]
        },
        {
            type: "select",
            label: "Slow MA",
            values: [5, 10, 14, 20, 25, 30, 50, 75, 100, 150, 200, 250]
        },
        {
            type: "select",
            label: "Condition",
            values: ["Less Than", "Greater Than", "Crossing Down", "Crossing Up"]
        },
        {
            type: "select",
            label: "Timeframe",
            values: ["1m", "5m", "15m", "1h", "4h", "1d"]
        }
    ]),

    BollingerBands: createConfig([
        {
            type: "select",
            label: "BB% Period",
            values: [10, 14, 20, 50, 100],
        },
        {
            type: "select",
            label: "Deviation",
            values: [1, 1.5, 2, 2.5, 3]
        },
        {
            type: "select",
            label: "Condition",
            values: ["Less Than", "Greater Than", "Crossing Down", "Crossing Up"]
        },
        {
            type: "select",
            label: "Timeframe",
            values: ["1m", "5m", "15m", "1h", "4h", "1d"]
        },
        {
            type: "number",
            label: "Signal Value",
            placeholder: "Enter Signal Value"
        }
    ]),

    MACD: createConfig([
    // Replace separate Fast/Slow/Signal inputs with one "MACD Preset" dropdown
    {
      type: "select",
      label: "MACD Preset",
      // EXACT combos from your Python code:
      values: [
        "12,26,9",  // The standard combo used in rating logic
        "6,20,9",
        "9,30,9",
        "10,26,9",
        "15,35,9",
        "18,40,9"
      ]
    },
    {
      type: "select",
      label: "MACD Trigger",
      values: ["Crossing Up", "Crossing Down"]
    },
    {
      type: "select",
      label: "Line Trigger",
      values: ["Less Than 0", "Greater Than 0"]
    },
    {
      type: "select",
      label: "Timeframe",
      values: ["1m", "5m", "15m", "1h", "4h", "1d"]
    }
  ]),

    Stochastic: createConfig([
    // Replace K Length, K Smoothing, D Smoothing with a single "Preset"
    {
      type: "select",
      label: "Stochastic Preset",
      // EXACT combos from your Python code: (k_len, k_smooth, d_smooth)
      values: [
        "14,3,3",
        "14,3,5",
        "20,5,5",
        "21,7,7",
        "28,9,9"
      ]
    },
    {
      type: "select",
      label: "K Condition",
      values: ["Less Than", "Greater Than", "Crossing Down", "Crossing Up"]
    },
    {
      type: "number",
      label: "K Signal Value",
      placeholder: "Enter K Signal Value"
    },
    {
      type: "select",
      label: "Condition",
      values: ["K Crossing Up D", "K Crossing Down D"]
    },
    {
      type: "select",
      label: "Timeframe",
      values: ["1m", "5m", "15m", "1h", "4h", "1d"]
    }
  ]),


    ParabolicSAR: createConfig([
    // Replace separate "Start" and "Maximum" with a single "PSAR Preset"
    {
      type: "select",
      label: "PSAR Preset",
      // EXACT combos from your Python code:
      values: [
        "0.02,0.2", // the original standard
        "0.03,0.2",
        "0.04,0.3",
        "0.05,0.4",
        "0.06,0.5"
      ]
    },
    {
      type: "select",
      label: "Condition",
      values: ["Crossing (Long)", "Crossing (Short)"]
    },
    {
      type: "select",
      label: "Timeframe",
      values: ["1m", "5m", "15m", "1h", "4h", "1d"]
    }
  ])
};



// Toggle Safety Orders
document.getElementById('safety-order-toggle').addEventListener('change', function () {
    const safetyOrderConfig = document.getElementById('safety-order-config');
    safetyOrderConfig.style.display = this.checked ? 'block' : 'none';
});

// Stop Loss Configuration Toggle
toggleElement(
    document.getElementById('stop-loss-toggle'),
    document.getElementById('stop-loss-config'),
    { showStyle: "block", hideStyle: "none" }
);

// Update Inputs on Indicator Selection
document.getElementById('indicator-select').addEventListener('change', function () {
    const selectedIndicator = this.value;
    const inputsContainer = document.getElementById('indicator-inputs');
    inputsContainer.innerHTML = indicatorConfigs[selectedIndicator] || '';
});

// Add Entry Conditions Dynamically
const addEntryCondition = document.getElementById('add-entry-condition');
const entryConditionGroup = document.getElementById('entry-condition-group');

addEntryCondition.addEventListener('click', () => {
    const newCondition = entryConditionGroup.firstElementChild.cloneNode(true);
    entryConditionGroup.appendChild(newCondition);
});

// Add Safety Order Conditions Dynamically
const addSafetyCondition = document.getElementById('add-safety-condition');
const safetyConditionGroup = document.getElementById('safety-condition-group');

addSafetyCondition.addEventListener('click', () => {
    const newCondition = safetyConditionGroup.firstElementChild.cloneNode(true);
    safetyConditionGroup.appendChild(newCondition);
});

// Add Safety Order Conditions Dynamically
const addExitCondition = document.getElementById('add-exit-condition');
const exitConditionGroup = document.getElementById('exit-condition-group');

addExitCondition.addEventListener('click', () => {
    const newCondition = exitConditionGroup.firstElementChild.cloneNode(true);
    exitConditionGroup.appendChild(newCondition);
});

// Load Cryptocurrency Pairs from Server
async function loadCryptoPairs() {
    const dropdownList = document.getElementById('dropdown-list');

    try {
        const response = await fetch('/get_crypto_pairs');
        const pairs = await response.json();

        pairs.forEach(pair => {
            const item = document.createElement('div');
            item.className = 'dropdown-item';
            item.innerHTML = `<input type="checkbox" value="${pair}"> ${pair}`;
            dropdownList.appendChild(item);
        });
    } catch (error) {
    }
}

// Load Crypto Pairs on Page Load
window.onload = loadCryptoPairs;


