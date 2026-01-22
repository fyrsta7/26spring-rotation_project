    }
    return copy_file(src_path, dst_path, src_stat, src_fd);
}

/**
 * Copy a source file to a destination file. Returns true if successful, false 
 * otherwise. If there is an error, its description is output to stderr.
 * 
 * To avoid repeated work, the source file's stat and file descriptor are required.
 */
bool copy_file(String src_path, String dst_path, struct stat src_stat, int src_fd)
{
    int dst_fd = creat(dst_path.characters(), 0666);
    if (dst_fd < 0) {
        if (errno != EISDIR) {
            perror("open dst");
            return false;
        }
        StringBuilder builder;
        builder.append(dst_path);
        builder.append('/');
        builder.append(FileSystemPath(src_path).basename());
        dst_path = builder.to_string();
        dst_fd = creat(dst_path.characters(), 0666);
        if (dst_fd < 0) {
            perror("open dst");
            return false;
        }
    }

    if (src_stat.st_size > 0) {
        // NOTE: This is primarily an optimization, so it's not the end if it fails.
        ftruncate(dst_fd, src_stat.st_size);
    }

    for (;;) {
        char buffer[32768];
        ssize_t nread = read(src_fd, buffer, sizeof(buffer));
        if (nread < 0) {
            perror("read src");
            return false;
        }
        if (nread == 0)
            break;
        ssize_t remaining_to_write = nread;
        char* bufptr = buffer;
        while (remaining_to_write) {
            ssize_t nwritten = write(dst_fd, bufptr, remaining_to_write);
            if (nwritten < 0) {
                perror("write dst");
                return false;
            }
            assert(nwritten > 0);
            remaining_to_write -= nwritten;
            bufptr += nwritten;
        }
    }

    auto my_umask = umask(0);
    umask(my_umask);
