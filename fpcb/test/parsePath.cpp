#include <string>
#include <vector>
#include <tchar.h>
#include <windows.h>
#include <shlwapi.h>
#include "parsePath.h"

static std::vector<std::string> split(const std::string &s, const std::string &seperator) 
{
	std::vector<std::string> result;
	typedef std::string::size_type string_size;
	string_size i = 0;

	while (i != s.size()) {
		//找到字符串中首个不等于分隔符的字母；
		int flag = 0;
		while (i != s.size() && flag == 0) {
			flag = 1;
			for (string_size x = 0; x < seperator.size(); ++x)
				if (s[i] == seperator[x]) {
					++i;
					flag = 0;
					break;
				}
		}

		//找到又一个分隔符，将两个分隔符之间的字符串取出；
		flag = 0;
		string_size j = i;
		while (j != s.size() && flag == 0) {
			for (string_size x = 0; x < seperator.size(); ++x)
				if (s[j] == seperator[x]) {
					flag = 1;
					break;
				}
			if (flag == 0)
				++j;
		}
		if (i != j) {
			result.push_back(s.substr(i, j - i));
			i = j;
		}
	}
	return result;
}

static std::string WcharToChar(const wchar_t* wp, size_t m_encode = CP_ACP)
{
	std::string str;
	int len = WideCharToMultiByte((int)m_encode, 0, wp, wcslen(wp), NULL, 0, NULL, NULL);
	char    *m_char = new char[len + 1];
	WideCharToMultiByte((int)m_encode, 0, wp, wcslen(wp), m_char, len, NULL, NULL);
	m_char[len] = '\0';
	str = m_char;
	delete m_char;
	return str;
}


static void getFiles(
	const char* path,
	const char* fileSuffix,
	std::vector<std::string> &files)
{
	// TODO
	char strTemp[2048];
	TCHAR tstrTemp[1024];
	std::string strPath = std::string(path);

	sprintf_s(strTemp, "%s\\*%s", path, fileSuffix);

	WIN32_FIND_DATA wfd;
	HANDLE hFind;

#ifdef UNICODE  
	MultiByteToWideChar(CP_ACP, 0, strTemp, -1, tstrTemp, 1024);
#else  
	strcpy(tstrTemp, strTemp);
#endif  
	int i = 0;

	files.clear();
	hFind = FindFirstFile(tstrTemp, &wfd);

	do
	{
		std::string strName;
		std::string strFile;

		strName = WcharToChar(wfd.cFileName);
		strFile = strPath + "\\" + strName;

		files.push_back(strFile);

	} while (FindNextFile(hFind, &wfd));
}


static std::string subContext(const std::string& s)
{
	int ipos = s.rfind("\\");
	std::string szTemp = s.substr(ipos + 1);
	ipos = szTemp.rfind(".");
	return szTemp.substr(0, ipos);
}

void paresPath(const std::string& root,
	std::vector< std::vector<std::string> >& frontPath,
	std::vector< std::vector<std::string> >& backPath,
    int imageNum)
{
	std::vector<std::string> path;
	getFiles(root.c_str(), ".jpg", path);

	std::vector<std::string> cpPath = path;

	std::vector< std::vector<std::string> > vecPath;
	for (auto it = cpPath.begin(); it != cpPath.end(); )
	{
		std::string szMomPath = *it;
		std::string szContext = *it;
		it = cpPath.erase(it);

		szContext = subContext(szContext);
		std::vector<std::string> szVec = split(szContext, "_");
		if (szVec.size() != 7)
			continue;

		int iCamCnt = imageNum;//atoi(szVec[4].c_str());
		int isFront = -1;
		if (szVec[6] == "r")
			isFront = 0;
		else if (szVec[6] == "f")
			isFront = 1;

		//find other path
		std::vector<std::string> pathGroup;
		pathGroup.emplace_back(szMomPath);

		for (int i = 0; i < cpPath.size(); i++)
		{
			std::string szTemp = subContext(cpPath[i]);
			std::vector<std::string> szVecTemp = split(szTemp, "_");
			if (szVecTemp.size() != 7)
				continue;
			if (szVecTemp[1] == szVec[1])
			{
				pathGroup.emplace_back(cpPath[i]);
				cpPath[i] = "";
			}
		}

		if (pathGroup.size() == iCamCnt)
		{
			if (isFront == 0)
				backPath.emplace_back(pathGroup);
			else if (isFront == 1)
				frontPath.emplace_back(pathGroup);
		}
	}

	return;
}
