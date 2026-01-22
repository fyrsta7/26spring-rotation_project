    set_callbacks (Open, Close);

    add_bool ("file-mmap", true, NULL,
              FILE_MMAP_TEXT, FILE_MMAP_LONGTEXT, true);
vlc_module_end();

static block_t *Block (access_t *);
static int Seek (access_t *, int64_t);
static int Control (access_t *, int, va_list);

struct access_sys_t
{
    size_t page_size;
    size_t mtu;
    int    fd;
};

#define MMAP_SIZE (1 << 20)

static int Open (vlc_object_t *p_this)
{
    access_t *p_access = (access_t *)p_this;
    access_sys_t *p_sys;
    const char *path = p_access->psz_path;
    int fd;

    if (!var_CreateGetBool (p_this, "file-mmap"))
        return VLC_EGENERIC; /* disabled */

    STANDARD_BLOCK_ACCESS_INIT;

    if (!strcmp (p_access->psz_path, "-"))
        fd = dup (0);
    else
    {
        msg_Dbg (p_access, "opening file %s", path);
        fd = utf8_open (path, O_RDONLY | O_NOCTTY, 0666);
    }

    if (fd == -1)
    {
        msg_Warn (p_access, "cannot open %s: %m", path);
        goto error;
    }

    /* mmap() is only safe for regular and block special files.
     * For other types, it may be some idiosyncrasic interface (e.g. packet
     * sockets are a good example), if it works at all. */
    struct stat st;

    if (fstat (fd, &st))
    {
        msg_Err (p_access, "cannot stat %s: %m", path);
        goto error;
    }

    if (!S_ISREG (st.st_mode) && !S_ISBLK (st.st_mode))
    {
        msg_Dbg (p_access, "skipping non regular file %s", path);
        goto error;
    }

    /* Autodetect mmap() support */
    if (st.st_size > 0)
    {
        void *addr = mmap (NULL, 1, PROT_READ|PROT_WRITE, MAP_PRIVATE, fd, 0);
        if (addr != MAP_FAILED)
            munmap (addr, 1);
        else
            goto error;
