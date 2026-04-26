self.importScripts('data/car-deals.js');
// Files to cache
var cacheName = 'car-deals-v1';
var appShellFiles = [
    './',
    'data/car-deals.js',
    'index.html',
    'scripts/car-logic.js',
    'style.css',
    'favicon.ico',
    'img/car-deals.png',
    'img/bg.png',
    'resources/material-design-lite/material.min.js',
    'resources/material-design-lite/material.red-indigo.min.css'
];

// Generate image paths for all cars in car-deals.js
const carImages = [
  'data/img/AudiAllroad2001.jpg',
  'data/img/KiaAmanti2004.jpg',
  'data/img/LexusRX1999.jpg',
  'data/img/ToyotaCressida1992.jpg'
];
// Combine core files and car images into one cache list
const contentToCache = appShellFiles.concat(carImages);

// INSTALL: Save all files into the browser cache
self.addEventListener('install', (e) => {
    e.waitUntil(
        caches.open(cacheName).then((cache) => {
            return cache.addAll(contentToCache);
    })
    );
});

// FETCH: Serve from cache first, then try the network
self.addEventListener('fetch', (e) => {
    e.respondWith(
        caches.match(e.request).then((response) => {
            return response || fetch(e.request).then((networkResponse) => {
                return caches.open(cacheName).then((cache) => {
                    // Save a copy of the new network response for next time
                    cache.put(e.request, networkResponse.clone());
                    return networkResponse;
                });
            });
        })
    );
});

// ACTIVATE: Delete old caches if the cacheName changes
self.addEventListener('activate', (e) => {
    e.waitUntil(
        caches.keys().then((keyList) => {
            return Promise.all(keyList.map((key) => {
                if (key !== cacheName) {
                    return caches.delete(key);
                }
            }));
        })
    );
});

