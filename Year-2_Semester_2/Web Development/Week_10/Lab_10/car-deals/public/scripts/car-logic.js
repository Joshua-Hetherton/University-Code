//Registering Service Worker
if ('serviceWorker' in navigator) {
 navigator.serviceWorker.register('sw.js');
}


// Function to swap placeholder for real car photo
function loadImages(img) {
    const realImage = img.getAttribute('data-src');

    if (realImage) {
        img.src = realImage;
        img.onload = () => {
            img.removeAttribute('data-src');
        };
    }
}

// Using Template Literals to populate car information and dialog details
const renderCar = (car, index) => `
<article onclick="document.querySelector('.${car.slug}').showModal();">
    <img src="data/img/placeholder.png" data-src="data/img/${car.slug}.jpg" alt="${car.name}">
    <h3>#${index + 1}. ${car.name}</h3>
    <ul>
        <li><span>Price:</span> <strong>${car.price}</strong></li>
        <li><span>Brand:</span> <strong>${car.brand}</strong></li>
        <li><span>Model:</span> <strong>${car.model}</strong></li>
        <li><span>Year:</span> <strong>${car.year}</strong></li>
    </ul>
</article>
<dialog class="mdl-dialog ${car.slug}">
    <h4 class="mdl-dialog__title">${car.name}</h4>
    <div class="mdl-dialog__content">
        <p>Type: ${car.type}</p>
        <p>Fuel type: ${car.fuel_type}</p>
        <p>Gear: ${car.gear}</p>
        <p>Mileage: ${car.mileage}</p>
        <p>Description: ${car.description}</p>
    </div>
    <div class="mdl-dialog__actions">
        <button type="button" class="mdl-button close"
            onclick="document.querySelector('.${car.slug}').close();">
            Close
        </button>
    </div>
</dialog>
`;

// Generate all car tiles + dialogs
let content = cars.map((car, index) => renderCar(car, index)).join('');

document.getElementById('content').innerHTML = content;

// Lazy-load images
let imagesToLoad = document.querySelectorAll('img[data-src]');

if ('IntersectionObserver' in window) {
    let observer = new IntersectionObserver((items, observer) => {
        items.forEach(item => {
            if (item.isIntersecting) {
                loadImages(item.target);
                observer.unobserve(item.target);
            }
        });
    });

    imagesToLoad.forEach(img => observer.observe(img));
} else {
    imagesToLoad.forEach(img => loadImages(img));
}
