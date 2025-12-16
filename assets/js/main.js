/**
* Template Name: Squadfree
* Template URL: https://bootstrapmade.com/squadfree-free-bootstrap-template-creative/
* Updated: Aug 07 2024 with Bootstrap v5.3.3
* Author: BootstrapMade.com
* License: https://bootstrapmade.com/license/
*/

(function() {
  "use strict";

  /**
   * Apply .scrolled class to the body as the page is scrolled down
   */
  function toggleScrolled() {
    const selectBody = document.querySelector('body');
    const selectHeader = document.querySelector('#header');
    if (!selectHeader.classList.contains('scroll-up-sticky') && !selectHeader.classList.contains('sticky-top') && !selectHeader.classList.contains('fixed-top')) return;
    window.scrollY > 100 ? selectBody.classList.add('scrolled') : selectBody.classList.remove('scrolled');
  }

  document.addEventListener('scroll', toggleScrolled);
  window.addEventListener('load', toggleScrolled);

  /**
   * Mobile nav toggle
   */
  const mobileNavToggleBtn = document.querySelector('.mobile-nav-toggle');

  function mobileNavToogle() {
    document.querySelector('body').classList.toggle('mobile-nav-active');
    mobileNavToggleBtn.classList.toggle('bi-list');
    mobileNavToggleBtn.classList.toggle('bi-x');
  }
  mobileNavToggleBtn.addEventListener('click', mobileNavToogle);

  /**
   * Hide mobile nav on same-page/hash links
   */
  document.querySelectorAll('#navmenu a').forEach(navmenu => {
    navmenu.addEventListener('click', () => {
      if (document.querySelector('.mobile-nav-active')) {
        mobileNavToogle();
      }
    });

  });

  /**
   * Toggle mobile nav dropdowns
   */
  document.querySelectorAll('.navmenu .toggle-dropdown').forEach(navmenu => {
    navmenu.addEventListener('click', function(e) {
      e.preventDefault();
      this.parentNode.classList.toggle('active');
      this.parentNode.nextElementSibling.classList.toggle('dropdown-active');
      e.stopImmediatePropagation();
    });
  });

  /**
   * Preloader
   */
  const preloader = document.querySelector('#preloader');
  if (preloader) {
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', () => {
        preloader.remove();
      });
    } else {
      // DOM 已经就绪，直接移除
      preloader.remove();
    }
  }

  /**
   * Scroll top button
   */
  let scrollTop = document.querySelector('.scroll-top');

  function toggleScrollTop() {
    if (scrollTop) {
      window.scrollY > 100 ? scrollTop.classList.add('active') : scrollTop.classList.remove('active');
    }
  }
  scrollTop.addEventListener('click', (e) => {
    e.preventDefault();
    window.scrollTo({
      top: 0,
      behavior: 'smooth'
    });
  });

  window.addEventListener('load', toggleScrollTop);
  document.addEventListener('scroll', toggleScrollTop);

  /**
   * Animation on scroll function and init
   */
  function aosInit() {
    AOS.init({
      duration: 600,
      easing: 'ease-in-out',
      once: true,
      mirror: false
    });
  }
  // 在 DOM 内容就绪后初始化 AOS，这样能和 preloader 同步
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', aosInit);
  } else {
    aosInit();
  }

  /**
   * Initiate Pure Counter
   */
  new PureCounter();

  /**
   * Initiate glightbox
   */
  const glightbox = GLightbox({
    selector: '.glightbox'
  });

  /**
   * Init isotope layout and filters
   */
  document.querySelectorAll('.isotope-layout').forEach(function(isotopeItem) {
    let layout = isotopeItem.getAttribute('data-layout') ?? 'masonry';
    let filter = isotopeItem.getAttribute('data-default-filter') ?? '*';
    let sort = isotopeItem.getAttribute('data-sort') ?? 'original-order';

    let initIsotope;
    imagesLoaded(isotopeItem.querySelector('.isotope-container'), function() {
      initIsotope = new Isotope(isotopeItem.querySelector('.isotope-container'), {
        itemSelector: '.isotope-item',
        layoutMode: layout,
        filter: filter,
        sortBy: sort
      });
    });

    isotopeItem.querySelectorAll('.isotope-filters li').forEach(function(filters) {
      filters.addEventListener('click', function() {
        isotopeItem.querySelector('.isotope-filters .filter-active').classList.remove('filter-active');
        this.classList.add('filter-active');
        initIsotope.arrange({
          filter: this.getAttribute('data-filter')
        });
        if (typeof aosInit === 'function') {
          aosInit();
        }
      }, false);
    });

  });

  /**
   * Init swiper sliders
   */
  function initSwiper() {
    document.querySelectorAll(".init-swiper").forEach(function(swiperElement) {
      let config = JSON.parse(
        swiperElement.querySelector(".swiper-config").innerHTML.trim()
      );

      if (swiperElement.classList.contains("swiper-tab")) {
        initSwiperWithCustomPagination(swiperElement, config);
      } else {
        new Swiper(swiperElement, config);
      }
    });
  }

  window.addEventListener("load", initSwiper);

  /**
   * Correct scrolling position upon page load for URLs containing hash links.
   */
  window.addEventListener('load', function(e) {
    if (window.location.hash) {
      if (document.querySelector(window.location.hash)) {
        setTimeout(() => {
          let section = document.querySelector(window.location.hash);
          let scrollMarginTop = getComputedStyle(section).scrollMarginTop;
          window.scrollTo({
            top: section.offsetTop - parseInt(scrollMarginTop),
            behavior: 'smooth'
          });
        }, 100);
      }
    }
  });

  /**
   * Navmenu Scrollspy
   */
  let navmenulinks = document.querySelectorAll('.navmenu a');

  function navmenuScrollspy() {
    navmenulinks.forEach(navmenulink => {
      if (!navmenulink.hash) return;
      let section = document.querySelector(navmenulink.hash);
      if (!section) return;
      let position = window.scrollY + 200;
      if (position >= section.offsetTop && position <= (section.offsetTop + section.offsetHeight)) {
        document.querySelectorAll('.navmenu a.active').forEach(link => link.classList.remove('active'));
        navmenulink.classList.add('active');
      } else {
        navmenulink.classList.remove('active');
      }
    })
  }
  window.addEventListener('load', navmenuScrollspy);
  document.addEventListener('scroll', navmenuScrollspy);

  /**
   * Lazy Loading for Images
   */
  function lazyLoadImages(sectionId) {
    const section = document.getElementById(sectionId);
    if (!section) return;
    
    const lazyImages = section.querySelectorAll('img[data-src]');
    
    lazyImages.forEach((img) => {
      const imageSrc = img.getAttribute('data-src');
      if (imageSrc) {
        const imageLoader = new Image();
        imageLoader.onload = function() {
          img.src = imageSrc;
          img.classList.add('loaded');
          img.removeAttribute('data-src');
          // 移除加载中的占位符
          const container = img.closest('.lazy-image-container');
          if (container) {
            container.classList.add('loaded');
            container.style.background = 'transparent';
          }
        };
        imageLoader.onerror = function() {
          img.classList.add('loaded');
          const container = img.closest('.lazy-image-container');
          if (container) {
            container.innerHTML = '<span style="color: #999;">图片加载失败</span>';
          }
        };
        imageLoader.src = imageSrc;
      }
    });
  }

  function lazyLoadPortfolioImages() {
    lazyLoadImages('portfolio');
  }

  // 等待 hero 背景图加载完，先加载 team section，最后加载 portfolio
  const heroBg = document.getElementById('hero-bg');
  if (heroBg) {
    const startImageLoad = function() {
      // 先加载 team section 的图片（优先级高）
      setTimeout(function() { lazyLoadImages('team'); }, 100);
      // 最后加载 portfolio 图片（优先级最低）
      setTimeout(lazyLoadPortfolioImages, 500);
    };

    if (heroBg.complete) {
      // 背景图已经在缓存中，直接开始加载
      startImageLoad();
    } else {
      // 背景图加载完成后再开始加载
      heroBg.addEventListener('load', startImageLoad, { once: true });
      // 如果背景图加载失败，也不要一直卡着
      heroBg.addEventListener('error', startImageLoad, { once: true });
    }
  } else {
    // 找不到 hero 背景图时的兜底处理：回退到 DOMContentLoaded 触发
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', function() {
        // 先加载 team section
        setTimeout(function() { lazyLoadImages('team'); }, 100);
        // 最后加载 portfolio
        setTimeout(lazyLoadPortfolioImages, 500);
      });
    } else {
      // 先加载 team section
      setTimeout(function() { lazyLoadImages('team'); }, 100);
      // 最后加载 portfolio
      setTimeout(lazyLoadPortfolioImages, 500);
    }
  }

})();