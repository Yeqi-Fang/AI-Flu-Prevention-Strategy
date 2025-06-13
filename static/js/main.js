// 全局函数和通用功能

$(document).ready(function() {
    // 初始化工具提示
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'))
    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl)
    });
    
    // 侧边栏高亮当前页面
    const currentPath = window.location.pathname;
    $('.sidebar .nav-link').each(function() {
        if ($(this).attr('href') === currentPath) {
            $(this).addClass('active');
        }
    });
    
    // 文件选择后显示文件名
    $('input[type="file"]').on('change', function() {
        const fileName = $(this).val().split('\\').pop();
        $(this).siblings('.form-text').remove();
        if (fileName) {
            $(this).after(`<small class="form-text text-success">已选择: ${fileName}</small>`);
        }
    });
    
    // 表单验证
    $('.needs-validation').on('submit', function(event) {
        if (!this.checkValidity()) {
            event.preventDefault();
            event.stopPropagation();
        }
        $(this).addClass('was-validated');
    });
    
    // 返回顶部按钮
    $(window).scroll(function() {
        if ($(this).scrollTop() > 100) {
            $('#backToTop').fadeIn();
        } else {
            $('#backToTop').fadeOut();
        }
    });
    
    $('#backToTop').click(function() {
        $('html, body').animate({scrollTop: 0}, 800);
        return false;
    });
});

// 显示加载动画
function showLoading(message = '加载中...') {
    const loadingHtml = `
        <div class="loading-overlay">
            <div class="spinner-border text-primary" role="status">
                <span class="visually-hidden">Loading...</span>
            </div>
            <p class="mt-2">${message}</p>
        </div>
    `;
    $('body').append(loadingHtml);
}

// 隐藏加载动画
function hideLoading() {
    $('.loading-overlay').remove();
}

// 显示成功提示
function showSuccess(message) {
    const alertHtml = `
        <div class="alert alert-success alert-dismissible fade show position-fixed top-0 end-0 m-3" role="alert">
            <i class="bi bi-check-circle"></i> ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        </div>
    `;
    $('body').append(alertHtml);
    
    // 3秒后自动消失
    setTimeout(function() {
        $('.alert-success').fadeOut(function() {
            $(this).remove();
        });
    }, 3000);
}

// 显示错误提示
function showError(message) {
    const alertHtml = `
        <div class="alert alert-danger alert-dismissible fade show position-fixed top-0 end-0 m-3" role="alert">
            <i class="bi bi-exclamation-circle"></i> ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        </div>
    `;
    $('body').append(alertHtml);
}

// 确认对话框
function confirmAction(message, callback) {
    if (confirm(message)) {
        callback();
    }
}

// 格式化数字（添加千分位）
function formatNumber(num) {
    return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}

// 下载文件
function downloadFile(url, filename) {
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
}

// AJAX 默认设置
$.ajaxSetup({
    beforeSend: function(xhr, settings) {
        // 添加 CSRF token（如果需要）
        if (!/^(GET|HEAD|OPTIONS|TRACE)$/i.test(settings.type) && !this.crossDomain) {
            xhr.setRequestHeader("X-CSRFToken", getCookie('csrf_token'));
        }
    },
    error: function(xhr, status, error) {
        console.error('AJAX Error:', status, error);
        hideLoading();
        
        let errorMessage = '请求失败，请稍后重试';
        if (xhr.responseJSON && xhr.responseJSON.error) {
            errorMessage = xhr.responseJSON.error;
        }
        showError(errorMessage);
    }
});

// 获取Cookie值
function getCookie(name) {
    let cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        const cookies = document.cookie.split(';');
        for (let i = 0; i < cookies.length; i++) {
            const cookie = cookies[i].trim();
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}

// 添加加载样式
const style = document.createElement('style');
style.textContent = `
    .loading-overlay {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(0, 0, 0, 0.5);
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        z-index: 9999;
        color: white;
    }
    
    #backToTop {
        position: fixed;
        bottom: 20px;
        right: 20px;
        display: none;
        background-color: #007bff;
        color: white;
        border: none;
        border-radius: 50%;
        width: 50px;
        height: 50px;
        text-align: center;
        line-height: 50px;
        font-size: 20px;
        cursor: pointer;
        box-shadow: 0 2px 5px rgba(0,0,0,0.3);
        z-index: 1000;
    }
    
    #backToTop:hover {
        background-color: #0056b3;
    }
`;
document.head.appendChild(style);

// 添加返回顶部按钮
$('body').append('<button id="backToTop" title="返回顶部"><i class="bi bi-arrow-up"></i></button>');