const CACHE_NAME = "eatlytic-cache-v1";
const ASSETS_TO_CACHE = [
  "/",
  "/static/index.html",
  "/static/style.css",
  "/static/about.html",
  "/static/privacy.html",
  "/static/terms.html",
  "/developer",
  "/static/developer.html"
];

// Install Event
self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      console.log("[Service Worker] Caching core assets");
      return cache.addAll(ASSETS_TO_CACHE);
    })
  );
  self.skipWaiting();
});

// Activate Event
self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) => {
      return Promise.all(
        keys.map((key) => {
          if (key !== CACHE_NAME) {
            console.log("[Service Worker] Removing old cache:", key);
            return caches.delete(key);
          }
        })
      );
    })
  );
  self.clients.claim();
});

// Fetch Event (Network First, then cache fallback for core pages)
self.addEventListener("fetch", (event) => {
  // Only handle GET requests
  if (event.request.method !== "GET") return;

  const url = new URL(event.request.url);

  // Skip API endpoints, Webhook routes, or dynamic search pages
  if (url.pathname.startsWith("/analyze") || 
      url.pathname.startsWith("/check-image") || 
      url.pathname.startsWith("/personalized") || 
      url.pathname.startsWith("/api") || 
      url.pathname.startsWith("/whatsapp") || 
      url.pathname.startsWith("/export-pdf") || 
      url.pathname.startsWith("/ingredients")) {
    return;
  }

  event.respondWith(
    fetch(event.request)
      .then((response) => {
        // If valid, clone and put it in cache
        if (response && response.status === 200 && response.type === "basic") {
          const responseClone = response.clone();
          caches.open(CACHE_NAME).then((cache) => {
            cache.put(event.request, responseClone);
          });
        }
        return response;
      })
      .catch(() => {
        // Network failed, serve from cache
        return caches.match(event.request);
      })
  );
});
