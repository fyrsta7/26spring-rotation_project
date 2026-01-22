            return TestResult::NotAFile;

        if (m_checkWritePermission && (fi.exists() && !fi.isWritable()))
            return TestResult::CantWrite;
        if (m_checkReadPermission && !fi.isReadable())
            return TestResult::CantRead;
    }

    return TestResult::OK;
}

Private::FileSystemPathValidator::TestResult Private::FileSystemPathValidator::lastTestResult() const
{
    return m_lastTestResult;
}

