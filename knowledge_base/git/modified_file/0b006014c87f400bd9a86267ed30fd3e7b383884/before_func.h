extern int auto_comment_line_char;

/* Windows only */
enum hide_dotfiles_type {
	HIDE_DOTFILES_FALSE = 0,
	HIDE_DOTFILES_TRUE,
	HIDE_DOTFILES_DOTGITONLY
};
extern enum hide_dotfiles_type hide_dotfiles;

enum log_refs_config {
