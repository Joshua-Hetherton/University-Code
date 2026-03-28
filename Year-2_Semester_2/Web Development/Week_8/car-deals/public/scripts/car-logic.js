// This function swaps the placeholder for the real car photo
function loadImages(img) {

    // 1. Get the real image path from the data-src attribute
    const realImage = img.getAttribute('data-src');

    // 2. If it exists, swap it into the src attribute
    if (realImage) {
        img.src = realImage;

        // 3. Once the high-res image loads, remove the data-src attribute
        // This triggers the CSS transition to remove the blur
        img.onload = () => {
            img.removeAttribute('data-src');
        };
    }
}

// Generating content based on the template
var template = "<article>\n\
<img src='data/img/placeholder.png' data-src='data/img/SLUG.jpg' alt='NAME'>\n\
<h3>#POS. NAME</h3>\n\
<ul>\n\
<li><span>Price:</span> <strong>PRICE</strong></li>\n\
<li><span>Brand:</span> <strong>BRAND</strong></li>\n\
<li><span>Model:</span> <strong>MODEL</strong></li>\n\
<li><span>Year:</span> <strong>YEAR</strong></li>\n\
</ul>\n\
</article>";

var content = '';

for (var i = 0; i < cars.length; i++) {

    // Below: Define a variable called *entry*. Replace the placeholders by pulling in cars data from the cardeals.js file
    var entry = template
        .replace(/POS/g, (i + 1))
        .replace(/SLUG/g, cars[i].slug)
        .replace(/NAME/g, cars[i].name)
        .replace(/PRICE/g, cars[i].price)
        .replace(/BRAND/g, cars[i].brand)
        .replace(/MODEL/g, cars[i].model)
        .replace(/YEAR/g, cars[i].year);

    entry = entry.replace('<a href=\'http://\'></a>', '-');

    // Adds the created entry variable to the content variable
    content += entry;
}

document.getElementById('content').innerHTML = content;
// Get the HTML element, ‘content’, and set its HTML content to the content variable created above

var imagesToLoad = document.querySelectorAll('img[data-src]');

if ('IntersectionObserver' in window) {

    var observer = new IntersectionObserver(function (items, observer) {

        items.forEach(function (item) {

            if (item.isIntersecting) {
                loadImages(item.target);
                observer.unobserve(item.target);
            }

        });

    });

    imagesToLoad.forEach(function (img) {
        observer.observe(img);
    });

} else {

    imagesToLoad.forEach(function (img) {

        loadImages(img);

    });

}