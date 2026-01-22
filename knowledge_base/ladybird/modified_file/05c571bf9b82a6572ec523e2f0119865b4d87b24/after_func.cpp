
namespace Web::HighResolutionTime {

JS_DEFINE_ALLOCATOR(Performance);

Performance::Performance(HTML::Window& window)
    : DOM::EventTarget(window.realm())
