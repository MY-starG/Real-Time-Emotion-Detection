function expandCardAndNavigate(card, url) {
  const overlay = document.getElementById('overlay');
  const rect = card.getBoundingClientRect();
  
  // Fade out and zoom the card content
  card.classList.add('expand');
  
  overlay.style.position = 'fixed';
  overlay.style.left = `${rect.left}px`;
  overlay.style.top = `${rect.top}px`;
  overlay.style.width = `${rect.width}px`;
  overlay.style.height = `${rect.height}px`;
  overlay.classList.add('expand');
  
  // Smoothly transition to fill the screen from the card
  setTimeout(() => {
      overlay.style.transition = 'all 0.5s ease-in-out';
      overlay.style.left = '0';
      overlay.style.top = '0';
      overlay.style.width = '100%';
      overlay.style.height = '100%';
  }, 10);
  
  // Update browser history and navigate to the new page
  setTimeout(() => {
      history.pushState({ url: window.location.href }, null, window.location.href);
      window.location.href = url;
  }, 500); // Adjust the duration to match the transition
}

// Handle browser back button
window.onpopstate = function(event) {
  if (event.state) {
      window.location.href = event.state.url;
  } else {
      window.location.href = '/'; // Main page URL
  }
};