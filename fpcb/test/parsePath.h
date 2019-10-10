#pragma once
#include <string>
#include <vector>

void paresPath(const std::string& root,
	std::vector< std::vector<std::string> >& frontPath,
	std::vector< std::vector<std::string> >& backPath,
    int imageNum);
