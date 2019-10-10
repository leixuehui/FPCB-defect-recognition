#include <sys/stat.h>
#include <io.h>
#include <direct.h>
#include "fpcbInspector.hpp"
#include "pugiconfig.hpp"
#include "pugixml.hpp"
//using namespace cv;

//#define OPENPREPROCESS
#define TEST 
#define USEING_DL  0 // in release mode,if using deep learning,     please set  1. 
				     //in release mode, if not using deep learning, please set  0.


Inspector::Inspector()
{

}

Inspector::~Inspector()
{

}


static std::mutex fmu;
static std::condition_variable condVar;
static bool directoryExist(std::string dir)
{
	struct stat status;
	if (stat(dir.c_str(), &status) != 0)	return false;
	return ((status.st_mode & S_IFDIR) != 0);
}

 //参数有点多

char gBuffer[256];
inline void wrtChl768f(pugi::xml_node &node, const std::string &childName, const float (&param)[768])
{
	node.append_child(childName.c_str());
	for (int i = 0; i < 768; i++)
	{
		sprintf(gBuffer, "%f", param[i]);
		node.child(childName.c_str()).append_child(pugi::node_pcdata).set_value(gBuffer);
		node.child(childName.c_str()).append_child(pugi::node_pcdata).set_value(",");
	}
}

inline void wrtChl3f(pugi::xml_node &node, const std::string &childName, const vec3f &param)
{
	node.append_child(childName.c_str());
	sprintf(gBuffer, "%f", param[0]);
	node.child(childName.c_str()).append_child(pugi::node_pcdata).set_value(gBuffer);
	node.child(childName.c_str()).append_child(pugi::node_pcdata).set_value(",");
	sprintf(gBuffer, "%f", param[1]);
	node.child(childName.c_str()).append_child(pugi::node_pcdata).set_value(gBuffer);
	node.child(childName.c_str()).append_child(pugi::node_pcdata).set_value(",");
	sprintf(gBuffer, "%f", param[2]);
	node.child(childName.c_str()).append_child(pugi::node_pcdata).set_value(gBuffer);
}

inline void wrtChl3f(pugi::xml_node &node, const std::string &childName, const vec3d &param)
{
	node.append_child(childName.c_str());
	sprintf(gBuffer, "%f", param[0]);
	std::cout << param[0] << std::endl;
	node.child(childName.c_str()).append_child(pugi::node_pcdata).set_value(gBuffer);
	node.child(childName.c_str()).append_child(pugi::node_pcdata).set_value(",");
	sprintf(gBuffer, "%f", param[1]);
	node.child(childName.c_str()).append_child(pugi::node_pcdata).set_value(gBuffer);
	node.child(childName.c_str()).append_child(pugi::node_pcdata).set_value(",");
	sprintf(gBuffer, "%f", param[2]);
	node.child(childName.c_str()).append_child(pugi::node_pcdata).set_value(gBuffer);
}

inline void wrtChl2i(pugi::xml_node &node, const std::string &childName, const cv::Size &param)
{
	node.append_child(childName.c_str());
	sprintf(gBuffer, "%d", param.width);
	node.child(childName.c_str()).append_child(pugi::node_pcdata).set_value(gBuffer);
	node.child(childName.c_str()).append_child(pugi::node_pcdata).set_value(",");
	sprintf(gBuffer, "%d", param.height);
	node.child(childName.c_str()).append_child(pugi::node_pcdata).set_value(gBuffer);
}

inline void wrtChi1i(pugi::xml_node &node, const std::string &childName, const int &param)
{
	node.append_child(childName.c_str());
	sprintf(gBuffer, "%d", param);
	node.child(childName.c_str()).append_child(pugi::node_pcdata).set_value(gBuffer);
	node.child(childName.c_str()).append_child(pugi::node_pcdata).set_value(",");
}

inline void rdChl768f(const pugi::xml_node &node, const std::string &childName, float (&param)[768])
{
	std::string paramStr = node.child(childName.c_str()).child_value();
	std::istringstream iss(paramStr);
	std::string temp;

	int i = 0;
	while (std::getline(iss, temp, ',') && i < 768)
	{
		param[i++] = std::stof(temp);
	}
}

inline void rdChl3f(const pugi::xml_node &node, const std::string &childName, vec3f &param)
{
	sscanf(node.child(childName.c_str()).child_value(), "%f,%f,%f", &param[0], &param[1], &param[2]);
}

inline void rdChl2i(const pugi::xml_node &node, const std::string &childName, int &width, int &height)
{
	sscanf(node.child(childName.c_str()).child_value(), "%d,%d", &width, &height);
}

inline void rdChl1i(const pugi::xml_node &node, const std::string &childName, int &param)
{
    sscanf(node.child(childName.c_str()).child_value(), "%d", &param);

}
inline int rdAttr1i(const pugi::xml_node &node, const std::string &AttrName)
{
	return node.attribute(AttrName.c_str()).as_int();
}

int Inspector::unifyConfigparam(int isTop, Layer layer)
{
    TrainParam *pTrainParam = NULL;

    if(isTop)   pTrainParam = &gTrainParamTop;
    else        pTrainParam = &gTrainParamBack;

    //寻找Layer对应的训练参数
    int layerIdx = pTrainParam->getSimpleIndex(layer);

    if (layerIdx == -1) return -1;

    for(int i = 0; i < pTrainParam->nSimple; i++)
    {
        for(int j = 0; j < pTrainParam->simpleParam[i].nItem; j++)
        {
            pTrainParam->simpleParam[i].data[j].unify = true;
        }
    }

    return 0;
}

int Inspector::getHistogram(const cv::Mat &image, int x, int y, uchar lower[3], uchar upper[3], float ch1[256], float ch2[256], float ch3[256])
{
    //初始化
    memset(lower, 0, sizeof(uchar) * 3);

    memset(upper, 255, sizeof(uchar) * 3);

    memset(ch1, 0, sizeof(float) * 256);

    memset(ch2, 0, sizeof(float) * 256);

    memset(ch3, 0, sizeof(float) * 256);

    memset(&itemSelected, 0, sizeof(OBJSELECTED));
    
    //无缺陷
    if (defectsForInterface.size() <= 0)    return -1;
    
    float minDist = 65535;

    float dist = 0;

    int idx = 0;    //目标缺陷

    int marginTop = 0, marginButtom = 0, marginLeft = 0, marginRight = 0;

    bool isGetObject = false;

    //定位缺陷
    for(int i = 0; i < defectsForInterface.size(); i++)
    {
        //查看是否在当前区域类
        marginTop = y - defectsForInterface[i].image[1];

        marginLeft = x - defectsForInterface[i].image[0];

        marginRight = defectsForInterface[i].image[0] + defectsForInterface[i].image[2] - x;

        marginButtom = defectsForInterface[i].image[1] + defectsForInterface[i].image[3] - y;

        if(marginTop < 0 || marginLeft < 0 || marginRight < 0 || marginButtom < 0)  continue;
        
        //获取点到边的最大距离
        dist =  __max(
                    __max(
                        __max(
                            marginLeft/*x*/, 
                            marginTop/*y*/
                            ),
                        marginRight
                        ),
                    marginButtom

                    );

        if (dist < minDist)
        {
            minDist = dist;

            idx = i;

            isGetObject = true;
        }
        
    }

    if (isGetObject == false)   return -1;
    
    itemSelected.isTop = defectsForInterface[idx].isTop;
    
    itemSelected.imageIdx = defectsForInterface[idx].n;

    itemSelected.pcsIdx = defectsForInterface[idx].image[4];

    itemSelected.layer = defectsForInterface[idx].layer;

    itemSelected.itemIdx = defectsForInterface[idx].itemIdx;

    itemSelected.getStatus = true;

    PCS_REGION *pRegion = &gItems[itemSelected.imageIdx][itemSelected.pcsIdx];

    ITEM_REGION *pItem;

    cv::Mat mask;

    //items的起始结束区域
    int sx = 0, sy = 0, ex = 0, ey = 0, width = 0, height = 0;

    cv::Rect roi, maskRoi;
    
    cv::Vec3b *pVec3b;
    
    uchar *pUchar;

    int area = 0;

    TrainParam *pTrainParam = NULL;

    MeanStd *pMeanStd = NULL;

    int simpleLayerIdx, complexLayerIdx;

    for(int i = 0; i < pRegion->itemsRegions.size(); i++)
    {
        if (pRegion->itemsRegions[i].type == itemSelected.layer)
        {
            //选中的为ID号，先将ID号转化为index
            int itemIdx = [pRegion,i](int x)
            {
                int nItem = pRegion->itemsRegions[i].items.size();

                for(int idx = 0; idx < nItem; idx++)
                {
                    if(x == pRegion->itemsRegions[i].items[idx].iID)
                    {
                        return idx;
                    }
                }
                return -1;
            }(itemSelected.itemIdx);
            
            if(itemIdx == -1)  continue;
            
            pItem = &pRegion->itemsRegions[i].items[itemIdx];

            cv::threshold(pItem->mask, mask, 200, 255, cv::THRESH_BINARY);

            sx = __max(0, pItem->iOffsetX);

            sy = __max(0, pItem->iOffsetY);

            ex = __min(pItem->iOffsetX + mask.cols, image.cols);

            ey = __min(pItem->iOffsetY + mask.rows, image.rows);

            width = ex - sx;

            height = ey - sy;

            roi = { sx, sy, width, height };

            if (ex <= 0 || width <= 0)	continue;

            maskRoi = { sx - pItem->iOffsetX, 0, width, height };

            for (int r = 0; r < height; r++)
            {
                pVec3b = image(roi).ptr<cv::Vec3b>(r);

                pUchar = mask(maskRoi).ptr<uchar>(r);

                for (int c = 0; c < width; c++)
                {
                    if(pUchar[c] > 0)
                    {
                        area++;

                        ch1[pVec3b[c][0]]++;

                        ch2[pVec3b[c][1]]++;

                        ch3[pVec3b[c][2]]++;
                    }
                }
            }
            
            if(area > 0)
            {
                for (int n = 0; n < 256; n++)
                {
                    ch1[n] /= area;

                    ch2[n] /= area;

                    ch3[n] /= area;
                }
            }
        }
    }
    
    if(itemSelected.isTop)
    {
        pTrainParam = &gTrainParamTop;
    }
    else
    {
        pTrainParam = &gTrainParamBack;
    }
    
    simpleLayerIdx = pTrainParam->getSimpleIndex(itemSelected.layer);
    
    complexLayerIdx = pTrainParam->getComplexParam(itemSelected.layer);

    //选中的缺陷为非可调节缺陷
    if (simpleLayerIdx == -1 && complexLayerIdx == -1)
    {
        itemSelected.getStatus = false;
        return -1;
    }

    if (simpleLayerIdx > complexLayerIdx)
    {
        pMeanStd = &pTrainParam->simpleParam[simpleLayerIdx].data[itemSelected.itemIdx];

        for(int i = 0; i < 3; i++)
        {
            lower[i] = __max(0, pMeanStd->mean[i] - pMeanStd->stddev[i] * pMeanStd->lower[i]);
            
            upper[i] = __min(255, pMeanStd->mean[i] + pMeanStd->stddev[i] * pMeanStd->upper[i]);
        }
        
    }

    return 0;
}

int Inspector::setLocalParam(uchar lower[3], uchar upper[3])
{
    if(itemSelected.getStatus == false) return -1;
    
    int isTop = 0;

    int imageIdx = 0;

    int pcsIdx = 0;

    int itemIdx = 0;

    Layer layer;

    isTop = itemSelected.isTop;

    imageIdx = itemSelected.imageIdx;

    pcsIdx = itemSelected.pcsIdx;
    
    itemIdx = itemSelected.itemIdx;

    layer = itemSelected.layer;

    TrainParam *pTrainParam;

    MeanStd *pMeanStd;

    if (isTop)  pTrainParam = &gTrainParamTop;
    else        pTrainParam = &gTrainParamBack;

    int simpleLayerIdx, complexLayerIdx;

    simpleLayerIdx = pTrainParam->getSimpleIndex(layer);

    complexLayerIdx = pTrainParam->getComplexParam(layer);

    if (simpleLayerIdx > complexLayerIdx)
    {
        pMeanStd = &pTrainParam->simpleParam[simpleLayerIdx].data[itemIdx];
        
        for(int c = 0; c < 3; c++)
        {
            pMeanStd->lower[c] = __max(0, (pMeanStd->mean[c] - lower[c])) / pMeanStd->stddev[c];

            pMeanStd->upper[c] = __max(0, (upper[c] - pMeanStd->mean[c])) / pMeanStd->stddev[c];
        }

        //修改参数后将检测统一设为false
        pMeanStd->unify = false;
    }

    writeXmlParam(gConfigPaths, *pTrainParam, isTop, -1);

    return 0;
}

int Inspector::readHyperParam(std::string path, HyperParam *param, int isTop)
{
	if (isTop)	path += "hyperParamFront.xml";
	else		path += "hyperParamBack.xml";

	pugi::xml_document doc;
	doc.load_file(path.c_str());

	if (doc.empty()) return -1;

	pugi::xml_node	paramNode;
	pugi::xml_node	swapNode;
	pugi::xml_node	padChild, steelChild, opacityChild,
					transparencyChild, charChild, holeChild,
					lineChild, fingerChild, carveChild,
					specifiedChild, ingoreChild;

	if (doc.child("HyperParam").empty())
	{
		//std::cout << __LINE__ << std::endl;
		return -2;
	}
	paramNode = doc.child("HyperParam");

#pragma region space
	rdChl2i(paramNode, "ImageSize", gImageWidth, gImageHeight);
	rdChl2i(paramNode, "maxPcsSize", gMaxPcsWidth, gMaxPcsHeight);
	if (gImageWidth <= 0 || gImageHeight <= 0 || gMaxPcsWidth <= 0 || gMaxPcsHeight <= 0)
	{
		
		//std::cout << __LINE__ <<  std::endl;
		//std::cout << gImageWidth<<" "<< gImageHeight<<" "<< gMaxPcsWidth <<" "<< gMaxPcsHeight << std::endl;
		return -2;
	}
#pragma endregion

#pragma region padChild

	padChild = paramNode.child("pad");
	if (!padChild.empty())
	{
		param->nPad = rdAttr1i(padChild, "nPad");
		
		for (int n = 0; n < param->nPad; n++)
		{
			std::string swapStr = "pad_" + std::to_string(n);
			swapNode = padChild.child(swapStr.c_str());

			
			if (swapNode.empty())
			{
				//std::cout << __LINE__ << "  N: " << n << std::endl;
				return -2;
			}
					

			rdChl3f(swapNode, "totalMean", param->pad[n].totalMean);
			rdChl3f(swapNode, "lowerMean", param->pad[n].lowerMean);
			rdChl3f(swapNode, "lowerStdDev", param->pad[n].lowerStdDev);
			rdChl3f(swapNode, "upperMean", param->pad[n].upperMean);
			rdChl3f(swapNode, "upperStdDev", param->pad[n].upperStdDev);
			rdChl768f(swapNode, "hist", param->pad[n].hist);

			rdChl3f(swapNode, "valueFeature", param->pad[n].valueFeature);
			rdChl3f(swapNode, "vectorFeature_0", param->pad[n].vectorFeature[0]);
			rdChl3f(swapNode, "vectorFeature_1", param->pad[n].vectorFeature[1]);
			rdChl3f(swapNode, "vectorFeature_2", param->pad[n].vectorFeature[2]);
			rdChl3f(swapNode, "totalMean_SGM", param->pad[n].totalMean_SGM);
		}
	}
#pragma endregion

#pragma region steelChild

	steelChild = paramNode.child("steel");
	if (!steelChild.empty())
	{
		param->nSteel = rdAttr1i(steelChild, "nSteel");

		for (int n = 0; n < param->nSteel; n++)
		{
			std::string swapStr = "steel_" + std::to_string(n);
			swapNode = steelChild.child(swapStr.c_str());
			
			if (swapNode.empty())
			{
				//std::cout << __LINE__ << "  N: " << n << std::endl;
				return -2;
			}
				

			rdChl3f(swapNode, "totalMean", param->steel[n].totalMean);
			rdChl3f(swapNode, "lowerMean", param->steel[n].lowerMean);
			rdChl3f(swapNode, "lowerStdDev", param->steel[n].lowerStdDev);
			rdChl3f(swapNode, "upperMean", param->steel[n].upperMean);
			rdChl3f(swapNode, "upperStdDev", param->steel[n].upperStdDev);
			rdChl768f(swapNode, "hist", param->steel[n].hist);
		}
	}
#pragma endregion

#pragma region opacityChild

	opacityChild = paramNode.child("opacity");
	if (!opacityChild.empty())
	{
		param->nOpacity = rdAttr1i(opacityChild, "nOpacity");

		for (int n = 0; n < param->nOpacity; n++)
		{
			std::string swapStr = "opacity_" + std::to_string(n);
			swapNode = opacityChild.child(swapStr.c_str());
			if (swapNode.empty())
			{
				//std::cout << __LINE__ << "  N: " << n << std::endl;
				return -2;
			}

			rdChl3f(swapNode, "totalMean", param->opacity[n].totalMean);
			rdChl3f(swapNode, "lowerMean", param->opacity[n].lowerMean);
			rdChl3f(swapNode, "lowerStdDev", param->opacity[n].lowerStdDev);
			rdChl3f(swapNode, "upperMean", param->opacity[n].upperMean);
			rdChl3f(swapNode, "upperStdDev", param->opacity[n].upperStdDev);
			rdChl768f(swapNode, "hist", param->opacity[n].hist);
		}
	}
#pragma endregion

#pragma region carveChild

    carveChild = paramNode.child("carve");
    if (!carveChild.empty())
    {
        param->nCarve = rdAttr1i(carveChild, "nCarve");

        for (int n = 0; n < param->nCarve; n++)
        {
            std::string swapStr = "carve_" + std::to_string(n);
            swapNode = carveChild.child(swapStr.c_str());
			if (swapNode.empty())
			{
				//std::cout << __LINE__ << "  N: " << n << std::endl;
				return -2;
			}

            rdChl3f(swapNode, "totalMean", param->carve[n].totalMean);
            rdChl3f(swapNode, "lowerMean", param->carve[n].lowerMean);
            rdChl3f(swapNode, "lowerStdDev", param->carve[n].lowerStdDev);
            rdChl3f(swapNode, "upperMean", param->carve[n].upperMean);
            rdChl3f(swapNode, "upperStdDev", param->carve[n].upperStdDev);
            rdChl768f(swapNode, "hist", param->carve[n].hist);
        }
    }
#pragma endregion

#pragma region transparencyChild

    transparencyChild = paramNode.child("transparency");
    if (!transparencyChild.empty())
    {
        param->nTransparency = rdAttr1i(transparencyChild, "nTransparency");

        for (int n = 0; n < param->nTransparency; n++)
        {
            std::string swapStr = "transparency_" + std::to_string(n);
            swapNode = transparencyChild.child(swapStr.c_str());
			if (swapNode.empty())
			{
				//std::cout << __LINE__ << "  N: " << n << std::endl;
				return -2;
			}
            rdChl3f(swapNode, "lower", param->transparency[n].lower);
            rdChl3f(swapNode, "upper", param->transparency[n].upper);
        }
    }
#pragma endregion

#pragma region holeChild

    holeChild = paramNode.child("hole");
    if (!holeChild.empty())
    {
        param->nCarve = rdAttr1i(holeChild, "nHole");

        for (int n = 0; n < param->nCarve; n++)
        {
            std::string swapStr = "hole_" + std::to_string(n);
            swapNode = holeChild.child(swapStr.c_str());
			if (swapNode.empty())
			{
				//std::cout << __LINE__ << "  N: " << n << std::endl;
				return -2;
			}
            param->hole[n].chioseChannel = rdAttr1i(swapNode, "channel");
        }
    }
#pragma endregion
	return 0;
}

int Inspector::writeHyperParam(std::string path, HyperParam *param, int isTop)
{
	if (isTop)	path += "hyperParamFront.xml";
	else		path += "hyperParamBack.xml";

	if (!directoryExist(path))
	{
		int err = remove(path.c_str());
		if (err == 0)
		{
			std::cout << "删除之前参数成功" << std::endl;
		}
		else
		{
			std::cout << "删除之前参数不成功，请在" << path << "路径下手动删除" << std::endl;
		}
		
	}
	pugi::xml_document doc;
	doc.load_file(path.c_str());
	pugi::xml_node	paramNode;
	pugi::xml_node	swapNode;
	pugi::xml_node	padChild, steelChild, opacityChild,
					transparencyChild, charChild, holeChild,
					lineChild, fingerChild, carveChild;

	if(doc.child("HyperParam").empty())	paramNode = doc.append_child("HyperParam");
	else								paramNode = doc.child("HyperParam");

#pragma region space
	wrtChl2i(paramNode, "ImageSize", cv::Size(gImageWidth, gImageHeight));
	wrtChl2i(paramNode, "maxPcsSize", cv::Size(gMaxPcsWidth, gMaxPcsHeight));
#pragma endregion

#pragma region padChild
	
	if (paramNode.child("pad").empty())
	{
		padChild = paramNode.append_child("pad");
		padChild.append_attribute("nPad");
		padChild.attribute("nPad").set_value(param->nPad);
	}

	for (int i = 0; i < param->nPad; i++)
	{
		std::string swapStr = ("pad_" + std::to_string(i)).c_str();
		if (paramNode.child(swapStr.c_str()).empty())	swapNode = padChild.append_child(swapStr.c_str());
		else											swapNode = padChild.child(swapStr.c_str());

		wrtChl3f(swapNode, "totalMean", param->pad[i].totalMean);
		wrtChl3f(swapNode, "lowerMean",	param->pad[i].lowerMean);
		wrtChl3f(swapNode, "lowerStdDev", param->pad[i].lowerStdDev);
		wrtChl3f(swapNode, "upperMean", param->pad[i].upperMean);
		wrtChl3f(swapNode, "upperStdDev", param->pad[i].upperStdDev);


		wrtChl3f(swapNode, "valueFeature", param->pad[i].valueFeature);
		wrtChl3f(swapNode, "vectorFeature_0", param->pad[i].vectorFeature[0]);
		wrtChl3f(swapNode, "vectorFeature_1", param->pad[i].vectorFeature[1]);
		wrtChl3f(swapNode, "vectorFeature_2", param->pad[i].vectorFeature[2]);
		wrtChl3f(swapNode, "totalMean_SGM", param->pad[i].totalMean_SGM);
		wrtChl768f(swapNode, "hist", param->pad[i].hist);
	}
#pragma endregion

#pragma region steelChild

	if (paramNode.child("steel").empty())
	{
		steelChild = paramNode.append_child("steel");
		steelChild.append_attribute("nSteel");
		steelChild.attribute("nSteel").set_value(param->nSteel);
	}

	for (int i = 0; i < param->nSteel; i++)
	{
		std::string swapStr = ("steel_" + std::to_string(i)).c_str();
		if (paramNode.child(swapStr.c_str()).empty())	swapNode = steelChild.append_child(swapStr.c_str());
		else											swapNode = steelChild.child(swapStr.c_str());

		wrtChl3f(swapNode, "totalMean", param->steel[i].totalMean);
		wrtChl3f(swapNode, "lowerMean", param->steel[i].lowerMean);
		wrtChl3f(swapNode, "lowerStdDev", param->steel[i].lowerStdDev);
		wrtChl3f(swapNode, "upperMean", param->steel[i].upperMean);
		wrtChl3f(swapNode, "upperStdDev", param->steel[i].upperStdDev);
		wrtChl768f(swapNode, "hist", param->steel[i].hist);
	}
#pragma endregion

#pragma region opacityChild

	if (paramNode.child("opacityChild").empty())
	{
		opacityChild = paramNode.append_child("opacity");
		opacityChild.append_attribute("nOpacity");
		opacityChild.attribute("nOpacity").set_value(param->nOpacity);
	}
	for (int i = 0; i < param->nOpacity; i++)
	{
		std::string swapStr = ("opacity_" + std::to_string(i)).c_str();
		if (paramNode.child(swapStr.c_str()).empty())	swapNode = opacityChild.append_child(swapStr.c_str());
		else											swapNode = opacityChild.child(swapStr.c_str());

		wrtChl3f(swapNode, "totalMean", param->opacity[i].totalMean);
		wrtChl3f(swapNode, "lowerMean", param->opacity[i].lowerMean);
		wrtChl3f(swapNode, "lowerStdDev", param->opacity[i].lowerStdDev);
		wrtChl3f(swapNode, "upperMean", param->opacity[i].upperMean);
		wrtChl3f(swapNode, "upperStdDev", param->opacity[i].upperStdDev);
		wrtChl768f(swapNode, "hist", param->opacity[i].hist);
	}
#pragma endregion

#pragma region carveChild

    if (paramNode.child("carveChild").empty())
    {
        carveChild = paramNode.append_child("carve");
        carveChild.append_attribute("nCarve");
        carveChild.attribute("nCarve").set_value(param->nCarve);
    }
    for (int i = 0; i < param->nCarve; i++)
    {
        std::string swapStr = ("carve_" + std::to_string(i)).c_str();
        if (paramNode.child(swapStr.c_str()).empty())	swapNode = carveChild.append_child(swapStr.c_str());
        else											swapNode = carveChild.child(swapStr.c_str());

        wrtChl3f(swapNode, "totalMean", param->carve[i].totalMean);
        wrtChl3f(swapNode, "lowerMean", param->carve[i].lowerMean);
        wrtChl3f(swapNode, "lowerStdDev", param->carve[i].lowerStdDev);
        wrtChl3f(swapNode, "upperMean", param->carve[i].upperMean);
        wrtChl3f(swapNode, "upperStdDev", param->carve[i].upperStdDev);
        wrtChl768f(swapNode, "hist", param->carve[i].hist);
    }
#pragma endregion

#pragma region transparencyChild

    if (paramNode.child("transparency").empty())
    {
        transparencyChild = paramNode.append_child("transparency");
        transparencyChild.append_attribute("nTransparency");
        transparencyChild.attribute("nTransparency").set_value(param->nTransparency);
    }
    for (int i = 0; i < param->nTransparency; i++)
    {
        std::string swapStr = ("transparency_" + std::to_string(i)).c_str();
        if (paramNode.child(swapStr.c_str()).empty())	swapNode = transparencyChild.append_child(swapStr.c_str());
        else											swapNode = transparencyChild.child(swapStr.c_str());

        wrtChl3f(swapNode, "lower", param->transparency[i].lower);
        wrtChl3f(swapNode, "upper", param->transparency[i].upper);
    }
#pragma endregion

#pragma region holeChild

    if (paramNode.child("holeChild").empty())
    {
        holeChild = paramNode.append_child("hole");
        holeChild.append_attribute("nHole");
        holeChild.attribute("nHole").set_value(param->nHole);
    }
    for (int i = 0; i < param->nHole; i++)
    {
        std::string swapStr = ("hole_" + std::to_string(i)).c_str();
        if (paramNode.child(swapStr.c_str()).empty())	swapNode = holeChild.append_child(swapStr.c_str());
        else											swapNode = holeChild.child(swapStr.c_str());

        swapNode.append_attribute("channel");
        swapNode.attribute("channel").set_value(param->hole[i].chioseChannel);
    }
#pragma endregion
	doc.save_file(path.c_str());
	return 0;
}

int Inspector::writeXmlParam(std::string path, const TrainParam & trainParam, int isTop, int nImage)
{
    std::string xmlFile = path + "TrainInfo.xml";

    pugi::xml_document doc;

    doc.load_file(path.c_str());

    if (doc.empty())    return -1;

    pugi::xml_node isTopNode, nImageNode, pcsIDNode, methodNode, layerNode, meanStdNode;

    //寻找正反面
    if (isTop)
    {
        isTopNode = doc.find_child_by_attribute("isTop", "name", "1");

        //存在的删除重新写入
        if(!isTopNode.empty())   doc.remove_child(isTopNode);

        isTopNode = doc.append_child("isTop");
        
        isTopNode.append_attribute("name");
        
        isTopNode.attribute("name").set_value("1");
        
        isTopNode.append_child("height");

        isTopNode.append_child("width");

        isTopNode.child("height").append_child(pugi::node_pcdata).set_value(std::to_string(trainParam.maxPcsHeight).c_str());

        isTopNode.child("width").append_child(pugi::node_pcdata).set_value(std::to_string(trainParam.maxPcsWidth).c_str());
    }
    else
    {
        isTopNode = doc.find_child_by_attribute("name", "0");

        if(!isTopNode.empty())  doc.remove_child(isTopNode);

        isTopNode = doc.append_child("isTop");

        isTopNode.append_attribute("name");

        isTopNode.attribute("name").set_value("0");

        isTopNode.append_child("height");

        isTopNode.append_child("width");

        isTopNode.child("height").append_child(pugi::node_pcdata).set_value(std::to_string(trainParam.maxPcsHeight).c_str());

        isTopNode.child("width").append_child(pugi::node_pcdata).set_value(std::to_string(trainParam.maxPcsWidth).c_str());
    }

    //寻找简单检测方案
    methodNode = isTopNode.find_child_by_attribute("method", "name", "simpleParam");

    //存在的删除重新写入
    if(!methodNode.empty())  isTopNode.remove_child(methodNode);

    methodNode = isTopNode.append_child("method");

    methodNode.append_attribute("name");

    methodNode.attribute("name").set_value("simpleParam");

    methodNode.append_attribute("number");

    methodNode.attribute("number").set_value(trainParam.nSimple);

    //寻找层级
    for(int n = 0; n < trainParam.nSimple; n++)
    {
        layerNode = methodNode.find_child_by_attribute("layer","name",std::to_string(trainParam.simpleParam[n].layer).c_str());

        //存在的删除重新写入
        if(!layerNode.empty())  methodNode.remove_child(layerNode);

        layerNode = methodNode.append_child("layer");

        layerNode.append_attribute("name");

        layerNode.attribute("name").set_value(trainParam.simpleParam[n].layer);

        layerNode.append_attribute("number");

        layerNode.attribute("number").set_value(trainParam.simpleParam[n].nItem);

        for(int k = 0; k <= trainParam.simpleParam[n].nItem; k++)
        {
            //读出参数
            meanStdNode = layerNode.find_child_by_attribute("meanstddev", "name", std::to_string(k).c_str());

            if (!meanStdNode.empty())    layerNode.remove_child(meanStdNode);

            meanStdNode = layerNode.append_child("meanstddev");

            meanStdNode.append_attribute("name");

            meanStdNode.attribute("name").set_value(k);

            //判断当前检测项是否为局部检测
            meanStdNode.append_attribute("unify");

            if(trainParam.simpleParam[n].data[k].unify)
            {
                meanStdNode.attribute("unify").set_value(1);
            }
            else
            {
                meanStdNode.attribute("unify").set_value(0);
            }

            std::string valueStr;

            for(int c = 0; c < 3; c++)
            {
                valueStr += std::to_string(trainParam.simpleParam[n].data[k].mean[c]) + "," + 
                            std::to_string(trainParam.simpleParam[n].data[k].stddev[c]) + "," + 
                            std::to_string(trainParam.simpleParam[n].data[k].lower[c]) + "," +
                            std::to_string(trainParam.simpleParam[n].data[k].upper[c]) + ";";
            }

            meanStdNode.append_child(pugi::node_pcdata).set_value(valueStr.c_str());
        }
    }

    //寻找复杂检测方案
    methodNode = isTopNode.find_child_by_attribute("method", "name", "complexParam");

    if(!methodNode.empty())  isTopNode.remove_child(methodNode);

    methodNode = isTopNode.append_child("method");

    methodNode.append_attribute("name");

    methodNode.attribute("name").set_value("complexParam");

    //寻找字符检测方案

    //寻找铜片检测方案

    doc.save_file(xmlFile.c_str());
    
    return 0;
}

int Inspector::readXmlParam(std::string path)
{
    std::string xmlFile = path + "TrainInfo.xml";

    pugi::xml_document doc;

    doc.load_file(xmlFile.c_str());

    if (doc.empty())    return -1;

    pugi::xml_node isTopNode, nImageNode, pcsIDNode, methodNode, layerNode, meanStdNode;

    TrainParam *pTrainParam = nullptr;

    for (isTopNode = doc.child("isTop"); isTopNode; isTopNode = isTopNode.next_sibling())
    {
        int isTop = isTopNode.attribute("name").as_int();

        if(isTop == 1)  pTrainParam = &gTrainParamTop;
        else            pTrainParam = &gTrainParamBack;

        pTrainParam->maxPcsHeight = atoi(isTopNode.child("height").child_value());

        pTrainParam->maxPcsWidth = atoi(isTopNode.child("width").child_value());

        gMaxPcsHeight = pTrainParam->maxPcsHeight;
        
        gMaxPcsWidth = pTrainParam->maxPcsWidth;

        for(methodNode = isTopNode.child("method"); methodNode; methodNode = methodNode.next_sibling())
        {
            std::string method = methodNode.attribute("name").as_string();

            int methodNum = methodNode.attribute("number").as_int();

            if(method == "simpleParam")
            {
                pTrainParam->nSimple = methodNum;

                int idx = 0;

                for(layerNode = methodNode.child("layer"); layerNode; layerNode = layerNode.next_sibling())
                {
                    Layer layer = (Layer)layerNode.attribute("name").as_int();

                    int layerNum = layerNode.attribute("number").as_int();

                    pTrainParam->simpleParam[idx].nItem = layerNum;

                    pTrainParam->simpleParam[idx].layer = layer;

                    for (meanStdNode = layerNode.child("meanstddev"); meanStdNode; meanStdNode = meanStdNode.next_sibling())
                    {
                        int layerID = meanStdNode.attribute("name").as_int();
                        
                        bool unify = meanStdNode.attribute("unify").as_bool();

                        std::string dataStr = meanStdNode.child_value();

                        std::string subData[3], val[4];

                        int semicolon = 0;

                        for(int i = 0; i < 3; i ++)
                        {
                            dataStr = dataStr.substr(semicolon);

                            int symbol = dataStr.find_first_of(';');
                            
                            subData[i] = dataStr.substr(0, symbol);
                            
                            int delim = 0;

                            for(int j = 0; j < 4; j++)
                            {
                                subData[i] = subData[i].substr(delim);

                                int delimPos = subData[i].find_first_of(',');

                                val[j] = subData[i].substr(0, delimPos);

                                delim = delimPos + 1;
                            }

                            pTrainParam->simpleParam[idx].data[layerID].mean[i] = atof(val[0].c_str());

                            pTrainParam->simpleParam[idx].data[layerID].stddev[i] = atof(val[1].c_str());

                            pTrainParam->simpleParam[idx].data[layerID].lower[i] = atof(val[2].c_str());

                            pTrainParam->simpleParam[idx].data[layerID].upper[i] = atof(val[3].c_str());

                            pTrainParam->simpleParam[idx].data[layerID].unify = unify;

                            semicolon = symbol + 1;
                        }
                    }
                    idx++;
                }
            }
            else
            {
                pTrainParam->nComplex = methodNum;
            }
        }
    }
    return 0;
}

int Inspector::epmFilter(const cv::Mat &src, cv::Mat &dst, int radius, float delta)
{
    cv::Mat data;
    cv::Mat meanI;
    cv::Mat varI;
    src.convertTo(data, CV_32FC3);
    data /= 255;
    int di = 2 * radius + 1;
    blur(data, meanI, cv::Size(di, di));
    blur(data.mul(data), varI, cv::Size(di, di));
    varI = varI - meanI.mul(meanI);
    data -= meanI;
    data = meanI + (varI.mul(data) / (varI + cv::Scalar::all(delta)));
    data *= 255;
    data.convertTo(dst, CV_8UC3);
    return 0;
}

int Inspector::init(const CoarseMatchData_Input matchData, int isTop)
{
	iniLogAlg();
	auto log = spdlog::get("log");

	//只初始化一次；
	static bool flagInit = true;
	if (flagInit)
	{
		std::string strFile(strrchr(__FILE__, '\\') + 1), strData(__TIMESTAMP__);
		log->info( "Current File: "+ strFile + " Compile Time:" + strData);

#ifdef _DEBUG
		log->info("Current Mode: DEBUG");
#else
		log->info("Current Mode: RELEASE: " + std::to_string(USEING_DL));
#endif
		flagInit = false;
	}
	
	gImageWidth = matchData.iImgWidth;
	gImageHeight = matchData.iImgHeight;
    //一张板有多少张图
	gImageNumber = matchData.iImgNumber;
    //当前图位置
    gAccImageNum = 0;
    
    //训练参数存储路径
	gConfigPaths = "";
    
	threadPool		= new ThreadPool(/*std::thread::hardware_concurrency() - 1*/);
   
    //定位结果
	gItems			= new ALL_REGION[gImageNumber];
    
    //训练参数
	memset(&gHyperParam, 0, sizeof(gHyperParam));

	//申请临时图像空间
	for (int n = 0; n < gImageNumber; n++)
	{
		gImageSpace.push_back(cv::Mat::zeros(cv::Size(gImageWidth, gImageHeight), CV_8UC3));
	}
	
	//申请Halcon处理空间
	//try
	//{
		int ret = initialSegment(matchData, isTop);
		if (ret != 0)	return -1;
	//}
	//catch (...)
	//{
	//	printf("初始化失败!\n");
	//	return -1;
	//}

#ifndef _DEBUG

#if USEING_DL
	//开启python通道
		AlgBase::cudaInit();
#endif
   
#endif // !_DEBUG

	gMaxPcsWidth = 0;
	gMaxPcsHeight = 0;
	return 0;
}

int Inspector::destory()
{
	//if (gImageSpace)	delete[] gImageSpace;
	//if (gImageMask)		delete[] gImageMask;
	if (threadPool)		delete		threadPool;
	if (gItems)			delete[]	gItems;
	//if (gParam)			delete		gParam;
	//if (gHyperParam)	delete[]	gHyperParam;
	//if (gTempSpace)		delete[]	gTempSpace;
	if (gMemoryPool)	free(gMemoryPool);
	/*for (int i = 0; i<256; i++)
	{
		if (gMatrixPadTrain[i].pMatrixTrain!=nullptr)
		{
			free(gMatrixPadTrain[i].pMatrixTrain);
			gMatrixPadTrain[i].pMatrixTrain = nullptr;
		}
	}*/
	return 0;
}

int Inspector::loadParam(std::string dir)
{
	gConfigPaths = dir;
    
    iniLogAlg();

    auto log = spdlog::get("log");

    if (gConfigPaths.empty())
    {
        log->error("未设置训练数据路径");
        return -1;
    }
    
    char endChar = *gConfigPaths.rbegin();

    if(endChar != '\/' || endChar != '\\')
    {
        gConfigPaths += "/";
    }
    else if(endChar == '\\')
    {
        endChar = *(gConfigPaths.rbegin() + 1);
        if(endChar != '\\')
        {
            gConfigPaths += "\\";
        }
    }
    
	if (!directoryExist(dir))
	{
		_mkdir(dir.c_str());
		return 0;
	}
    readXmlParam(dir);
    
	//int err = 0;
	//err = readHyperParam(dir, &gHyperParam, 1);
	//switch (err)
	//{
	//case -1:
	//	std::cout << "正面未训练" << std::endl;
	//	break;
	//case -2:
	//	std::cout << "未获取正面正常参数" << std::endl;
	//	break;
	//default:
	//	AlgBase::loadParam(gHyperParam, 1);
	//	bPosParam = true;
	//	break;
	//};
	//err = readHyperParam(dir, &gHyperParam, 0);
	//switch (err)
	//{
	//case -1:
	//	std::cout << "正面未训练" << std::endl;
	//	break;
	//case -2:
	//	std::cout << "未获取反面正常参数" << std::endl;
	//	break;
	//default:
	//	AlgBase::loadParam(gHyperParam, 0);
	//	bNegParam = true;
	//	break;
	//};
    
	AlgBase::baseInit(std::thread::hardware_concurrency() - 1, cv::Size(gMaxPcsWidth, gMaxPcsHeight), cv::Size(gImageWidth, gImageHeight));
	
	return 0;
}

int Inspector::learn(cv::Mat &learnImage, int imageIndex, int isTop)
{
	
	iniLogAlg();
	auto log = spdlog::get("log");

	std::string strFunction(__FUNCTION__);

	log->info("========"+ strFunction +"========");
	log->info("1 输入检查 2 图像定位分割 3 参数训练/整理 4 参数保存 ");

	
	log->info("********1/4 输入检查 (" + std::to_string(imageIndex) + "/" + std::to_string(gImageNumber) + ")********");
	if (gConfigPaths.empty())
	{
		log->error("参数配置路径有误！");
		return -1;
	}
    
	cv::Mat imgSegement;
	int err = 0;

	log->info("********2/4 图像定位分割 (" + std::to_string(imageIndex) + "/" + std::to_string(gImageNumber) + ")********");
	
	cv::cvtColor(learnImage, imgSegement, cv::ColorConversionCodes::COLOR_BGR2RGB);
	try
	{
#ifndef OPENPREPROCESS
        learnImage.copyTo(gImageSpace[imageIndex]);
#else
        //cv::LUT(learnImage, lut, gImageSpace[imageIndex]);

        //epmFilter(learnImage, gImageSpace[imageIndex], 5, 0.01);
#endif
		err = doSegment(imgSegement, imageIndex, isTop, gItems[imageIndex]);
		if (err != 0)
		{
			log->error("图像定位分割失败");
			return -3;
		}
	}
	catch (...)
	{
		log->error("图像定位分割失败");
		return -3;
	}

	//重新初始化参数
	log->info("*********3/4 参数训练/整理 (" + std::to_string(imageIndex) + "/" + std::to_string(gImageNumber) + ")********");
	
	
	//cv::Mat test = learnImage.clone();
	//hoLocation(learnImage, test, 1);

	//std::vector<std::future<int>> vecFunc;
	log->info("3.1 参数训练");
    try
	{
		err = AlgBase::itemTrainerV2(learnImage, isTop, imageIndex, gItems[imageIndex], gTrainParam);
        
        //err = AlgBase::itemView(gConfigPaths,imgSegement, gItems[imageIndex], gHyperParam);
		if (err != 0)
		{
			log->error("参数训练失败");
			return -3;
		}
	
	}
	catch (...)
	{
		log->error("参数训练失败");
		return -3;
	}

	log->info("3.2 参数加载");

    gImageSpace[imageIndex] = learnImage.clone();
	//memcpy(&gParam, &_param, sizeof(ConfigParam));
	log->info("参数加载完毕");

	
	//训练完后初始化空间
	if (imageIndex == gImageNumber - 1)
	{
        gMaxPcsHeight = gTrainParam.maxPcsHeight;

        gMaxPcsWidth = gTrainParam.maxPcsWidth;

        AlgBase::baseInit(std::thread::hardware_concurrency() - 1, cv::Size(gMaxPcsWidth, gMaxPcsHeight), cv::Size(gImageWidth, gImageHeight));

		log->info("训练数据整理");

        log->info("简单训练数据整理");

        for (int i = 0; i < gTrainParam.nSimple; i++)
        {
            int nItem = gTrainParam.simpleParam[i].nItem;

            for(int j = 0; j < nItem; j++)
            {
                for(int k = 0; k < 3; k++)
                {
                    gTrainParam.simpleParam[i].data[j].mean[k] *= PIXEL_NORMALIZATION;

                    gTrainParam.simpleParam[i].data[j].stddev[k] *= PIXEL_NORMALIZATION;

                    gTrainParam.simpleParam[i].data[nItem].mean[k] += gTrainParam.simpleParam[i].data[j].mean[k];

                    gTrainParam.simpleParam[i].data[nItem].stddev[k] += gTrainParam.simpleParam[i].data[j].stddev[k];
                }
            }

            //整体异色推荐
            for (int k = 0; k < 3; k++)
            {
                gTrainParam.simpleParam[i].data[nItem].mean[k] /= nItem;

                gTrainParam.simpleParam[i].data[nItem].stddev[k] /= nItem;
            }

        }
        
        log->info("简单训练数据整理完毕");

        log->info("复杂训练数据整理");
        
        for (int i = 0; i < gTrainParam.nComplex; i++)
        {
            
        }

        log->info("复杂训练数据整理完毕");

		log->info("训练数据整理完毕");

		log->info("*********4/4 参数保存 (" + std::to_string(imageIndex) + "/" + std::to_string(gImageNumber) + ")********");

        if (isTop)   memcpy(&gTrainParamTop, &gTrainParam, sizeof(TrainParam));
        else        memcpy(&gTrainParamBack, &gTrainParam, sizeof(TrainParam));

        writeXmlParam(gConfigPaths, gTrainParam, isTop, imageIndex);
		//writeHyperParam(gConfigPaths, &gHyperParam,isTop);
		//AlgBase::loadParam(gHyperParam, isTop);
		if(isTop)	bPosParam = true;
		else		bNegParam = true;
		log->info("参数保存完毕");
	}
	
	log->info("========参数学习算法完毕 (" + std::to_string(imageIndex) + "/" + std::to_string(gImageNumber) + ")========");
 	return 0;
}

int Inspector::setConfigparam(const ConfigParam &configParam, int isTop)
{
    SimpleParam *pSimpleParam;

    ComplexParam *pComplexParam;

    TrainParam *pTrainParam;

    iniLogAlg();

    auto log = spdlog::get("log");
    
    if (gConfigPaths.empty())
    {
        log->error("未设置训练数据路径");
        return -1;
    }

    int nItem = 0;
    
    if(isTop)
    {
        log->info("开始写入正面信息");

        memcpy(&gParamTop, &configParam, sizeof(ConfigParam));
        
        pTrainParam = &gTrainParamTop;
    }
    else
    {
        log->info("开始写入反面信息");

        memcpy(&gParamBack, &configParam, sizeof(ConfigParam));

        pTrainParam = &gTrainParamBack;
    }

    for (int i = 0; i < pTrainParam->nSimple; i++)
    {
        pSimpleParam = &pTrainParam->simpleParam[i];

        nItem = pTrainParam->simpleParam[i].nItem;

        switch (pSimpleParam->layer)
        {
        case lineLay_pad:
            for (int c = 0; c < 3; c++)
            {
                //相当于整体换成configParam。通过setLocalParam来局部操作
                pSimpleParam->data[nItem].lower[c] = __max(0, (pSimpleParam->data[nItem].mean[c] - configParam.PadParam.colorParam.lowerLimit[c])) / pSimpleParam->data[nItem].stddev[c];

                pSimpleParam->data[nItem].upper[c] = __max(0, (configParam.PadParam.colorParam.upperLimit[c] - pSimpleParam->data[nItem].mean[c])) / pSimpleParam->data[nItem].stddev[c];
            }

            break;

        case stiffenerLay_steel:
            for (int c = 0; c < 3; c++)
            {
                pSimpleParam->data[nItem].lower[c] = __max(0, (pSimpleParam->data[nItem].mean[c] - configParam.SteelParam.colorParam.lowerLimit[c])) / pSimpleParam->data[nItem].stddev[c];

                pSimpleParam->data[nItem].upper[c] = __max(0, (configParam.SteelParam.colorParam.upperLimit[c] - pSimpleParam->data[nItem].mean[c])) / pSimpleParam->data[nItem].stddev[c];
            }

            break;

        case printLay_EMI:
            for (int c = 0; c < 3; c++)
            {
                pSimpleParam->data[nItem].lower[c] = __max(0, (pSimpleParam->data[nItem].mean[c] - configParam.OpacityParam.colorParam.lowerLimit[c])) / pSimpleParam->data[nItem].stddev[c];

                pSimpleParam->data[nItem].upper[c] = __max(0, (configParam.OpacityParam.colorParam.upperLimit[c] - pSimpleParam->data[nItem].mean[c])) / pSimpleParam->data[nItem].stddev[c];
            }

            break;

        case carveLay:
            for (int c = 0; c < 3; c++)
            {
                pSimpleParam->data[nItem].lower[c] = __max(0, (pSimpleParam->data[nItem].mean[c] - configParam.CarveParam.colorParam.lowerLimit[c])) / pSimpleParam->data[nItem].stddev[c];

                pSimpleParam->data[nItem].upper[c] = __max(0, (configParam.CarveParam.colorParam.upperLimit[c] - pSimpleParam->data[nItem].mean[c])) / pSimpleParam->data[nItem].stddev[c];
            }

            break;

        default:
            break;
        }
    }

    for (int i = 0; i < pTrainParam->nComplex; i++)
    {

    }

    writeXmlParam(gConfigPaths, *pTrainParam, isTop, -1);

    log->info("信息写入完毕");

	return 0;
}

int Inspector::getConfigparam(ConfigParam &configParam, int isTop)
{

    ConfigParam defaultParam;

    SimpleParam *pSimpleParam;
    
    ComplexParam *pComplexParam;

    int nItem = 0;
    
    memcpy(&configParam, &defaultParam, sizeof(ConfigParam));
    
    iniLogAlg();

    auto log = spdlog::get("log");
    
    TrainParam *pTrainParam = NULL;

    if (isTop)
    {
        pTrainParam = &gTrainParamTop;
        if (gTrainParamTop.empty())
        {
            log->info("未寻找到训练过的正面资料，正面资料设为默认值");
        }
    }
    else
    {
        pTrainParam = &gTrainParamBack;
        if (gTrainParamBack.empty())
        {
            log->info("未寻找到训练过的反面资料，反面资料设为默认值");
        }
    }

    for (int i = 0; i < pTrainParam->nSimple; i++)
    {
        pSimpleParam = &pTrainParam->simpleParam[i];

        nItem = pTrainParam->simpleParam[i].nItem;

        switch (pSimpleParam->layer)
        {
        case lineLay_pad:
            for (int c = 0; c < 3; c++)
            {
                configParam.PadParam.colorParam.lowerLimit[c] = __max(0, pSimpleParam->data[nItem].mean[c] - pSimpleParam->data[nItem].stddev[c] * pSimpleParam->data[nItem].lower[c]);

                configParam.PadParam.colorParam.upperLimit[c] = __min(255, pSimpleParam->data[nItem].mean[c] + pSimpleParam->data[nItem].stddev[c] * pSimpleParam->data[nItem].upper[c]);
            }

            break;

        case stiffenerLay_steel:
            for (int c = 0; c < 3; c++)
            {
                configParam.SteelParam.colorParam.lowerLimit[c] = __max(0, pSimpleParam->data[nItem].mean[c] - pSimpleParam->data[nItem].stddev[c] * pSimpleParam->data[nItem].lower[c]);

                configParam.SteelParam.colorParam.upperLimit[c] = __min(255, pSimpleParam->data[nItem].mean[c] + pSimpleParam->data[nItem].stddev[c] * pSimpleParam->data[nItem].upper[c]);
            }

            break;

        case printLay_EMI:
            for (int c = 0; c < 3; c++)
            {
                configParam.OpacityParam.colorParam.lowerLimit[c] = __max(0, pSimpleParam->data[nItem].mean[c] - pSimpleParam->data[nItem].stddev[c] * pSimpleParam->data[nItem].lower[c]);

                configParam.OpacityParam.colorParam.upperLimit[c] = __min(255, pSimpleParam->data[nItem].mean[c] + pSimpleParam->data[nItem].stddev[c] * pSimpleParam->data[nItem].upper[c]);
            }

            break;

        case carveLay:
            for (int c = 0; c < 3; c++)
            {
                configParam.CarveParam.colorParam.lowerLimit[c] = __max(0, pSimpleParam->data[nItem].mean[c] - pSimpleParam->data[nItem].stddev[c] * pSimpleParam->data[nItem].lower[c]);

                configParam.CarveParam.colorParam.upperLimit[c] = __min(255, pSimpleParam->data[nItem].mean[c] + pSimpleParam->data[nItem].stddev[c] * pSimpleParam->data[nItem].upper[c]);
            }

            break;

        default:
            break;
        }
    }

    for (int i = 0; i < pTrainParam->nComplex; i++)
    {

    }

    setConfigparam(configParam, isTop);

	return 0;
}

int Inspector::loadSpace(OutputInfo *outputSpace)
{
	iniLogAlg();
	auto log = spdlog::get("log");

	std::string strFunction(__FUNCTION__);

	//log->info("========" + strFunction + "========");

	if (outputSpace == nullptr || outputSpace->image.ptr == nullptr)	return -1;
	float factor = outputSpace->scale;
	int outputWidth = outputSpace->image.width;
	int outputHeight = outputSpace->image.height;
	//检测申请空间是否满足
	int space = gImageWidth * gImageHeight * gImageNumber * factor * factor;

	/*log->info("?′?ó2?êy?ì2é￡o?í" + std::to_string(outputWidth) +
		" ??￡o" + std::to_string(outputHeight) +
		" ??·?òò×ó￡o" + std::to_string(factor));
	log->info("3?ê??ˉ2?êy?ì2é￡o?í" + std::to_string(gImageWidth) + 
		" ??￡o" + std::to_string(gImageHeight)+
		" êyá?￡o" + std::to_string(gImageNumber));*/
	if (space > outputWidth * outputHeight)
	{
		log->info("申请空间承载不了"+std::to_string(gImageNumber)+"张图像的拼接，至少需要空间为"+std::to_string((space + 1)/1024/1024)+" M字节！");
		//printf("申请空间承载不了%d张图像的拼接，至少需要空间为%d", gImageNumber, space + 1);
		return -2;
	}
	
	outputWidth *= factor;
	outputHeight *= factor;
	
	outputSpace->image.width = BYTEALIGNING(outputWidth);
	outputSpace->image.step = outputSpace->image.width * 3;
	outputSpace->image.height = outputHeight;

	gOutputInfo = outputSpace;
	return 0;
}

int Inspector::saveInputParam(const std::string &dir,const int isTop)
{
	std::ofstream adjuctableParam(dir, std::ios::out | std::ios::app);

	SYSTEMTIME timeLocal;
	GetLocalTime(&timeLocal);

	char data[36] = { 0 };
	sprintf(data, "%04d-%02d-%02d  %02d:%02d:%02d:%02d",
		timeLocal.wYear, timeLocal.wMonth, timeLocal.wDay,
		timeLocal.wHour, timeLocal.wMinute, timeLocal.wSecond, timeLocal.wMilliseconds);
	std::string strData = std::string(data);

	if (!adjuctableParam.is_open())
	{
		return -1;
	}
	if (isTop)
	{
		adjuctableParam << "========?y???éμ÷?ú?ì2a2?êy±í========" << std::endl;
	}
	else
	{
		adjuctableParam << "========·′???éμ÷?ú?ì2a2?êy±í========" << std::endl;
	}
	adjuctableParam << "Time Point:"<<std::right << std::setw(36) << strData << std::endl;

	adjuctableParam << "o??ì?ì2a2?êy(PadParam)￡o" << std::endl;
	adjuctableParam << "ê?·??ì2a￡o" << gParam.PadParam.vaild << std::endl;
	adjuctableParam << "ê?·?ê1ó?é??è?§?°￡o" << gParam.PadParam.usingDL << std::endl;
	adjuctableParam << "??2?é??Tòìé?￡o" << gParam.PadParam.colorParam.upperLimit[0] << " " << gParam.PadParam.colorParam.upperLimit[1] << " " << gParam.PadParam.colorParam.upperLimit[2] << std::endl;
	adjuctableParam << "??2????T?Tòìé?￡o" << gParam.PadParam.colorParam.lowerLimit[0] << " " << gParam.PadParam.colorParam.lowerLimit[1] << " " << gParam.PadParam.colorParam.lowerLimit[2] << std::endl;
	adjuctableParam << "×?D?è±?Y???y?D?μ￡o" << gParam.PadParam.colorParam.infArea << std::endl;
	adjuctableParam << "×?D?è±?YX?á?D?μ￡o" << gParam.PadParam.colorParam.infWidth << std::endl;
	adjuctableParam << "×?D?è±?YY?á?D?μ￡o" << gParam.PadParam.colorParam.infHeight << std::endl;
	adjuctableParam << "±??μ??D?????êy￡o" << gParam.PadParam.shrinkSize << std::endl;

	adjuctableParam << "±??μèúo??àà?￡o" << gParam.PadParam.ruseDist << std::endl;
	adjuctableParam << "é??è?§?°?D?μ￡o" << gParam.PadParam.cfg << std::endl;
	adjuctableParam << std::endl;



	adjuctableParam << "?????ì2a2?êy(SteelParam)￡o" << std::endl;
	adjuctableParam << "ê?·??ì2a￡o" << gParam.SteelParam.vaild << std::endl;
	adjuctableParam << "ê?·?ê1ó?é??è?§?°￡o" << gParam.SteelParam.usingDL << std::endl;
	adjuctableParam << "??2?é??Tòìé?￡o" << gParam.SteelParam.colorParam.upperLimit[0] << " " << gParam.SteelParam.colorParam.upperLimit[1] << " " << gParam.SteelParam.colorParam.upperLimit[2] << std::endl;
	adjuctableParam << "??2????T?Tòìé?￡o" << gParam.SteelParam.colorParam.lowerLimit[0] << " " << gParam.SteelParam.colorParam.lowerLimit[1] << " " << gParam.SteelParam.colorParam.lowerLimit[2] << std::endl;
	adjuctableParam << "×?D?è±?Y???y?D?μ￡o" << gParam.SteelParam.colorParam.infArea << std::endl;
	adjuctableParam << "×?D?è±?YX?á?D?μ￡o" << gParam.SteelParam.colorParam.infWidth << std::endl;
	adjuctableParam << "×?D?è±?YY?á?D?μ￡o" << gParam.SteelParam.colorParam.infHeight << std::endl;
	adjuctableParam << "±??μ??D?????êy￡o" << gParam.SteelParam.shrinkSize << std::endl;

	adjuctableParam << "è±?Y???y?ó3é￡o" << gParam.SteelParam.areaWeight << std::endl;
	adjuctableParam << "±??μèúo??àà?￡o" << gParam.SteelParam.ruseDist << std::endl;
	adjuctableParam << "é??è?§?°?D?μ￡o" << gParam.SteelParam.cfg << std::endl;
	adjuctableParam << std::endl;


	adjuctableParam << "2?í??÷°ü·a?ì2a2?êy(OpacityParam)￡o" << std::endl;
	adjuctableParam << "ê?·??ì2a￡o" << gParam.OpacityParam.vaild << std::endl;
	adjuctableParam << "ê?·?ê1ó?é??è?§?°￡o" << gParam.OpacityParam.usingDL << std::endl;
	adjuctableParam << "??2?é??Tòìé?￡o" << gParam.OpacityParam.colorParam.upperLimit[0] << " " << gParam.OpacityParam.colorParam.upperLimit[1] << " " << gParam.OpacityParam.colorParam.upperLimit[2] << std::endl;
	adjuctableParam << "??2????T?Tòìé?￡o" << gParam.OpacityParam.colorParam.lowerLimit[0] << " " << gParam.OpacityParam.colorParam.lowerLimit[1] << " " << gParam.OpacityParam.colorParam.lowerLimit[2] << std::endl;
	adjuctableParam << "×?D?è±?Y???y?D?μ￡o" << gParam.OpacityParam.colorParam.infArea << std::endl;
	adjuctableParam << "×?D?è±?YX?á?D?μ￡o" << gParam.OpacityParam.colorParam.infWidth << std::endl;
	adjuctableParam << "×?D?è±?YY?á?D?μ￡o" << gParam.OpacityParam.colorParam.infHeight << std::endl;
	adjuctableParam << "±??μ??D?????êy￡o" << gParam.OpacityParam.shrinkSize << std::endl;

	adjuctableParam << "±??μèúo??àà?￡o" << gParam.OpacityParam.ruseDist << std::endl;
	adjuctableParam << "é??è?§?°?D?μ￡o" << gParam.OpacityParam.cfg << std::endl;
	adjuctableParam << std::endl;



	adjuctableParam << "í??÷°ü·a?ì2a2?êy(TransprencyRegion)￡o" << std::endl;
	adjuctableParam << "ê?·??ì2a￡o" << gParam.TransprencyParam.vaild << std::endl;
	adjuctableParam << "ê?·?ê1ó?é??è?§?°￡o" << gParam.TransprencyParam.usingDL << std::endl;
	adjuctableParam << "°μ??óòèYDí?è￡o" << gParam.TransprencyParam.lowerTolerance << std::endl;
	adjuctableParam << "áá??óòèYDí?è￡o" << gParam.TransprencyParam.upperTolerance << std::endl;
	adjuctableParam << "×?D?è±?Y???y?D?μ￡o" << gParam.TransprencyParam.infArea << std::endl;

	adjuctableParam << "×?D?è±?YX?á?D?μ￡o" << gParam.TransprencyParam.infWidth << std::endl;
	adjuctableParam << "×?D?è±?YY?á?D?μ￡o" << gParam.TransprencyParam.infHeight << std::endl;
	adjuctableParam << "±??μ??D?????êy￡o" << gParam.TransprencyParam.shrinkSize << std::endl;
	adjuctableParam << "±??μèúo??àà?￡o" << gParam.TransprencyParam.ruseDist << std::endl;
	adjuctableParam << "é??è?§?°?D?μ￡o" << gParam.TransprencyParam.cfg << std::endl;
	adjuctableParam << std::endl;



	adjuctableParam << "?×?′?ì2a2?êy(HoleRegion)￡o" << std::endl;
	adjuctableParam << "ê?·??ì2a￡o" << gParam.HoleParam.vaild << std::endl;
	adjuctableParam << "ê?·?ê1ó?é??è?§?°￡¨?T￡?￡o" << std::endl;
	adjuctableParam << "±??μ??D?????êy￡o" << gParam.HoleParam.shrinkSize << std::endl;
	adjuctableParam << "±??μèúo??àà?￡o" << gParam.HoleParam.ruseDist << std::endl;
	adjuctableParam << "é??è?§?°?D?μ￡o" << gParam.HoleParam.cfg << std::endl;
	adjuctableParam << std::endl;



	adjuctableParam << "×?·??ì2a2?êy(CharRegion)￡o" << std::endl;
	adjuctableParam << "ê?·??ì2a￡¨?T￡?￡o" << gParam.CharParam.vaild << std::endl;
	adjuctableParam << "ê?·?ê1ó?é??è?§?°￡¨?T￡?￡o" << std::endl;
	adjuctableParam << "1ì?¨×?·?è±?Yμ?×?D????y￡o" << gParam.CharParam.infArea << std::endl;
	adjuctableParam << "±??μ??D?????êy￡o" << gParam.CharParam.shrinkSize << std::endl;
	adjuctableParam << "±??μèúo??àà?￡o" << gParam.CharParam.ruseDist << std::endl;

	adjuctableParam << "é??è?§?°?D?μ￡o" << gParam.CharParam.cfg << std::endl;
	adjuctableParam << std::endl;




	adjuctableParam << "???·2??ì2a2?êy(LineRegion)￡o" << std::endl;
	adjuctableParam << "ê?·??ì2a￡o" << gParam.LineParam.vaild << std::endl;
	adjuctableParam << "ê?·?ê1ó?é??è?§?°￡o" << gParam.LineParam.usingDL << std::endl;
	adjuctableParam << "±??μ??D?????êy￡o" << gParam.LineParam.shrinkSize << std::endl;
	adjuctableParam << "±??μèúo??àà?￡o" << gParam.LineParam.ruseDist << std::endl;
	adjuctableParam << "é??è?§?°?D?μ￡o" << gParam.LineParam.cfg << std::endl;
	adjuctableParam << std::endl;



	adjuctableParam << "??ê????ì2a2?êy(FingerParam)￡o" << std::endl;
	adjuctableParam << "ê?·??ì2a￡o" << gParam.FingerParam.vaild << std::endl;
	adjuctableParam << "ê?·?ê1ó?é??è?§?°￡o" << gParam.FingerParam.usingDL << std::endl;
	adjuctableParam << "±??μ??D?????êy￡o" << gParam.FingerParam.shrinkSize << std::endl;
	adjuctableParam << "±??μèúo??àà?￡o" << gParam.FingerParam.ruseDist << std::endl;
	adjuctableParam << "é??è?§?°?D?μ￡o" << gParam.FingerParam.cfg << std::endl;
	adjuctableParam << std::endl;



	adjuctableParam << "3??D2??ì2a2?êy(CarveRegion)￡o" << std::endl;
	adjuctableParam << "ê?·??ì2a￡o" << gParam.CarveParam.vaild << std::endl;
	adjuctableParam << "ê?·?ê1ó?é??è?§?°￡o" << gParam.CarveParam.usingDL << std::endl;
	adjuctableParam << "??2?é??Tòìé?￡o" << gParam.CarveParam.colorParam.upperLimit[0] << " " << gParam.CarveParam.colorParam.upperLimit[1] << " " << gParam.CarveParam.colorParam.upperLimit[2] << std::endl;
	adjuctableParam << "??2????T?Tòìé?￡o" << gParam.CarveParam.colorParam.lowerLimit[0] << " " << gParam.CarveParam.colorParam.lowerLimit[1] << " " << gParam.CarveParam.colorParam.lowerLimit[2] << std::endl;
	adjuctableParam << "×?D?è±?Y???y?D?μ￡o" << gParam.CarveParam.colorParam.infArea << std::endl;
	adjuctableParam << "×?D?è±?YX?á?D?μ￡o" << gParam.CarveParam.colorParam.infWidth << std::endl;
	adjuctableParam << "×?D?è±?YY?á?D?μ￡o" << gParam.CarveParam.colorParam.infHeight << std::endl;
	adjuctableParam << "±??μ??D?????êy￡o" << gParam.CarveParam.shrinkSize << std::endl;

	adjuctableParam << "±??μèúo??àà?￡o" << gParam.CarveParam.ruseDist << std::endl;
	adjuctableParam << "é??è?§?°?D?μ￡o" << gParam.CarveParam.cfg << std::endl;
	adjuctableParam << std::endl;



	adjuctableParam << "ì?êa??óò(SpecifiedRegion)￡o" << std::endl;
	adjuctableParam << "ê?·??ì2a(?T)￡o" << std::endl;
	adjuctableParam << "ê?·?ê1ó?é??è?§?°￡o" << gParam.specifiedParam.usingDL << std::endl;
	adjuctableParam << "??2?é??Tòìé?￡o" << gParam.specifiedParam.colorParam.upperLimit[0] << " " << gParam.specifiedParam.colorParam.upperLimit[1] << " " << gParam.specifiedParam.colorParam.upperLimit[2] << std::endl;
	adjuctableParam << "??2????T?Tòìé?￡o" << gParam.specifiedParam.colorParam.lowerLimit[0] << " " << gParam.specifiedParam.colorParam.lowerLimit[1] << " " << gParam.specifiedParam.colorParam.lowerLimit[2] << std::endl;
	adjuctableParam << "×?D?è±?Y???y?D?μ￡o" << gParam.specifiedParam.colorParam.infArea << std::endl;
	adjuctableParam << "×?D?è±?YX?á?D?μ￡o" << gParam.specifiedParam.colorParam.infWidth << std::endl;
	adjuctableParam << "×?D?è±?YY?á?D?μ￡o" << gParam.specifiedParam.colorParam.infHeight << std::endl;
	adjuctableParam << "±??μèúo??àà?￡o" << gParam.specifiedParam.ruseDist << std::endl;
	adjuctableParam << "é??è?§?°?D?μ￡o" << gParam.specifiedParam.cfg << std::endl;
	adjuctableParam << std::endl;

	adjuctableParam << std::endl;
	adjuctableParam << "================================" << std::endl;

}

//software new mode;
int Inspector::process(cv::Mat &input, int imageIndex, int isTop)
{
#ifdef _TIME_LOG_
	std::chrono::steady_clock::time_point timeStart = std::chrono::steady_clock::now();
	static int time1CastFull = 0;
	static int time2CastFull = 0;
	static int time3CastFull = 0;
	static int time4CastFull = 0;
	static int time5CastFull = 0;
#endif

	//std::cout << "----"<<log_time_name << std::endl;
	iniLogAlg();
	auto log = spdlog::get("log");

	std::string strFunction(__FUNCTION__);

	log->info("========"+ strFunction +"========");

	log->info("1 输入检查 2 图像定位分割 3 缺陷检测 4 缺陷数据打包 5 缺陷图像整理");
	log->info("********1/5 输入检查(" + std::to_string(imageIndex + 1) + " / " + std::to_string(gImageNumber) + ")*******");
	if (isTop && gTrainParamTop.empty())
	{
		log->error("未训练正面参数，或正面参数训练有误！");
		return -1;
	}
	if (!isTop && gTrainParamBack.empty())
	{
		log->error("未训练反面参数，或反面参数训练有误！");
		return -1;
	}
	if (gConfigPaths.empty())
	{
		log->error("参数配置路径有误！");
		return -1;
	}

	if (gOutputInfo == nullptr)
	{
		log->error("没有检测到目标输出空间.");
		return -1;
	}

    TrainParam *pTrainParam;
    ConfigParam *pConfigParam;

    if(isTop)
    {
        pTrainParam = &gTrainParamTop;
        pConfigParam = &gParamTop;
    }
    else
    {
        pTrainParam = &gTrainParamBack;
        pConfigParam = &gParamBack;
    }

	int err = 0;

	//临时
	
	{

////    焊盘
// 	gParam.PadParam.colorParam.infArea = 15;
// 	gParam.PadParam.colorParam.infWidth = 10;
// 	gParam.PadParam.colorParam.infHeight = 10;
// 	gParam.PadParam.colorParam.upperLimit[0] = 12.f;
// 	gParam.PadParam.colorParam.upperLimit[1] = 12.f;
// 	gParam.PadParam.colorParam.upperLimit[2] = 12.f;
// 	gParam.PadParam.colorParam.lowerLimit[0] = 1.5f;
// 	gParam.PadParam.colorParam.lowerLimit[1] = 1.5f;
// 	gParam.PadParam.colorParam.lowerLimit[2] = 1.5f;
//// 	//钢片
// 	gParam.SteelParam.colorParam.infArea = 40;
// 	gParam.SteelParam.colorParam.infWidth = 30;
// 	gParam.SteelParam.colorParam.infHeight = 30;
// 	gParam.SteelParam.colorParam.upperLimit[0] = 4.f;
// 	gParam.SteelParam.colorParam.upperLimit[1] = 4.f;
// 	gParam.SteelParam.colorParam.upperLimit[2] = 4.f;
// 	gParam.SteelParam.colorParam.lowerLimit[0] = 2.f;
// 	gParam.SteelParam.colorParam.lowerLimit[1] = 2.f;
// 	gParam.SteelParam.colorParam.lowerLimit[2] = 2.f;
//// 	//黑封
// 	gParam.OpacityParam.colorParam.infArea = 30;
// 	gParam.OpacityParam.colorParam.infWidth = 30;
// 	gParam.OpacityParam.colorParam.infHeight = 30;
// 	gParam.OpacityParam.colorParam.upperLimit[0] = 15.f;
// 	gParam.OpacityParam.colorParam.upperLimit[1] = 15.f;
// 	gParam.OpacityParam.colorParam.upperLimit[2] = 15.f;
// 	gParam.OpacityParam.colorParam.lowerLimit[0] = 30.f;
// 	gParam.OpacityParam.colorParam.lowerLimit[1] = 30.f;
// 	gParam.OpacityParam.colorParam.lowerLimit[2] = 30.f;
//// 	//冲切
// 	gParam.CarveParam.colorParam.infArea = 40;
// 	gParam.CarveParam.colorParam.infWidth = 30;
// 	gParam.CarveParam.colorParam.infHeight = 30;
// 	gParam.CarveParam.colorParam.upperLimit[0] = 15.f;
// 	gParam.CarveParam.colorParam.upperLimit[1] = 15.f;
// 	gParam.CarveParam.colorParam.upperLimit[2] = 15.f;
// 	gParam.CarveParam.colorParam.lowerLimit[0] = 30.f;
// 	gParam.CarveParam.colorParam.lowerLimit[1] = 30.f;
// 	gParam.CarveParam.colorParam.lowerLimit[2] = 30.f;

	}
	
	if (isTop)
	{
		err = saveInputParam(".\\"+ log_param_name, 1);
		if (err == -1)
		{
			log->error("?y??ê?è?2?êy±￡′?òì3￡?￡");
		}
	}
	if (!isTop)
	{
		err = saveInputParam(".\\"+ log_param_name, 0);
		if (err == -1)
		{
			log->error("±3??ê?è?2?êy±￡′?òì3￡?￡");
		}
	}
	
#ifdef _TIME_LOG_
	std::chrono::steady_clock::time_point time1 = std::chrono::steady_clock::now();
	std::ofstream file(log_time_name, std::ios::out | std::ios::app);
	std::chrono::milliseconds timeCastSingle = std::chrono::duration_cast<std::chrono::milliseconds>(time1 - timeStart);
	std::chrono::milliseconds timeCastFull = std::chrono::duration_cast<std::chrono::milliseconds>(time1 - timeStart);
	time1CastFull += timeCastSingle.count();
	file <<"  "<<std::left << std::setw(32)<<strFunction +"("+ std::to_string(imageIndex + 1) + "/" + std::to_string(gImageNumber)+")1 输入检查"
		<< std::right << std::setw(16) << timeCastFull.count()
		<< std::right << std::setw(16) << timeCastSingle.count() 
		<< std::right << std::setw(16) << time1CastFull << std::endl;
#endif

	log->info("********2/5 图像定位分割 (" + std::to_string(imageIndex + 1 ) + " / " + std::to_string(gImageNumber) + ")*******");
	cv::Mat imgSegement;
	cv::cvtColor(input, imgSegement, cv::ColorConversionCodes::COLOR_BGR2RGB);
	try
	{
#ifndef OPENPREPROCESS
		input.copyTo(gImageSpace[imageIndex]);
#else
		//cv::LUT(input, lut, gImageSpace[imageIndex]);
		//epmFilter(learnImage, gImageSpace[imageIndex], 5, 0.01);
#endif
		//cv::cvtColor(input, gImageSpace[imageIndex], cv::COLOR_RGB2BGR);
		
		err = doSegment(imgSegement, imageIndex, isTop, gItems[imageIndex]);

		if (err != 0)
		{
			log->error("分割失败!");
			return -3;
		}
	}
	catch (...)
	{
		log->error("分割失败!");
		return -3;
	}


#ifdef _TIME_LOG_
	std::chrono::steady_clock::time_point time2 = std::chrono::steady_clock::now();
	timeCastSingle = std::chrono::duration_cast<std::chrono::milliseconds>(time2 - time1);
	timeCastFull = std::chrono::duration_cast<std::chrono::milliseconds>(time2 - timeStart);
	time2CastFull += timeCastSingle.count();
	//std::ofstream file(log_time_name, std::ios::out | std::ios::app);
	file <<"  "<<std::left << std::setw(32) << strFunction + "("+std::to_string(imageIndex + 1) + "/" + std::to_string(gImageNumber)+")2 图像定位"
		<< std::right << std::setw(16) << timeCastFull.count()
		<< std::right << std::setw(16) << timeCastSingle.count() 
		<< std::right << std::setw(16) << time2CastFull << std::endl;
#endif

	log->info("********3/5 缺陷检测 (" + std::to_string(imageIndex + 1 ) + " / " + std::to_string(gImageNumber) + ")*******");
	std::vector<std::future<int>> vecFunc;
	double start = cv::getTickCount();

	int nPcs = gItems[imageIndex].size();

	std::vector<DefectInfo> defectList(nPcs);
	std::vector<DefectCudaInfo> defectListCuda(nPcs);
#ifndef _DEBUG

#if USEING_DL

	input.copyTo(gImageSpace[imageIndex]);

	log->info("CUDA Defects Inspect Alg Start. (" + std::to_string(imageIndex +1 ) + " / " + std::to_string(gImageNumber) + ")");

	vecFunc.emplace_back(threadPool->submit(
		AlgBase::itemInspectorCuda,
		&input,
		std::ref(gItems[imageIndex]),
		gParam,
		std::ref(defectListCuda)
	));

	/*AlgBase::itemInspectorCuda(
		&input,
		gItems[imageIndex],
		gParam,
		defectListCuda);*/
	log->info("CUDA Defects Inspect Alg End. (" + std::to_string(imageIndex + 1) + " / " + std::to_string(gImageNumber)+")");

	
#endif

#endif

	log->info("Commom Defects Inspect Alg Start.");
	for (int n = 0; n < nPcs; n++)
	{
		defectList[n].n = imageIndex;
		defectList[n].nPcs = n;

		/*log->info("Single Thread Mode.");
		log->info("PCS " + std::to_string(n) + "/" + std::to_string(nPcs) + " Start.");
		AlgBase::itemInspector(
			&input,
			isTop,
			gItems[imageIndex][n],
			gParam,
			defectList[n],
			true);
		log->info("PCS " + std::to_string(n) + "/" + std::to_string(nPcs) + " End.");*/

		log->info("Mulit Thread Mode.");
		log->info("PCS " + std::to_string(n + 1) + "/" + std::to_string(nPcs) + " Start.");
		vecFunc.emplace_back(threadPool->submit(
			AlgBase::itemInspectorV2,
			std::ref(input),
			std::ref(isTop),
            std::ref(imageIndex),
			std::ref(gItems[imageIndex][n]),
            std::ref(*pTrainParam),
			std::ref(*pConfigParam),
			std::ref(defectList[n]))
		);
        //AlgBase::itemInspectorV2(input,isTop,imageIndex, gItems[imageIndex][n], *pTrainParam, *pConfigParam, defectList[n]);
		log->info("PCS " + std::to_string(n + 1) + "/" + std::to_string(nPcs) + " End.");
	}

	for (int v = 0; v < vecFunc.size(); v++)
	{
		vecFunc[v].get();
	}

	log->info("Commom  Defects Inspect Alg End.");
	log->info("Alg Time:" + std::to_string((cv::getTickCount() - start) * 1000 / cv::getTickFrequency()) + " ms.");

#ifdef _TIME_LOG_
	std::chrono::steady_clock::time_point time3 = std::chrono::steady_clock::now();
	timeCastSingle = std::chrono::duration_cast<std::chrono::milliseconds>(time3 - time2);
	timeCastFull = std::chrono::duration_cast<std::chrono::milliseconds>(time3 - timeStart);
	//std::ofstream file(log_time_name, std::ios::out | std::ios::app);
	time3CastFull += timeCastSingle.count();
	file <<"  "<< std::left << std::setw(32) << strFunction + "("+std::to_string(imageIndex + 1) + "/" + std::to_string(gImageNumber)+")3 缺陷检测"
		<< std::right << std::setw(16) << timeCastFull.count()
		<< std::right << std::setw(16) << timeCastSingle.count() 
		<< std::right << std::setw(16) << time3CastFull << std::endl;
#endif

	log->info("********4 缺陷数据打包 " + std::to_string(imageIndex + 1) + " / " + std::to_string(gImageNumber) + "*******");
	int numDefectCommon = 0;
	for (int n = 0; n < defectList.size(); n++)
	{
		for (int i = 0; i < defectList[n].roi.size(); i++)
		{
			Layer abstract = defectList[n].roi[i].abstract;

			for (int j=0;j<defectList[n].roi[i].info.size();j++)
			{
				DefectRoi dftROI;
				dftROI.image[0] = defectList[n].roi[i].info[j].rect.x;
				dftROI.image[1] = defectList[n].roi[i].info[j].rect.y;
				dftROI.image[2] = defectList[n].roi[i].info[j].rect.width;
				dftROI.image[3] = defectList[n].roi[i].info[j].rect.height;
				dftROI.image[4] = defectList[n].nPcs;
				img2gerb(dftROI.image[1], dftROI.image[0],
					imageIndex, isTop,
					dftROI.gerber[1], dftROI.gerber[0]);
				img2gerb(dftROI.image[1] + dftROI.image[3],
					dftROI.image[0] + dftROI.image[2],
					imageIndex, isTop,
					dftROI.gerber[3], dftROI.gerber[2]);
				dftROI.gerber[2] -= dftROI.gerber[0];
				dftROI.gerber[3] -= dftROI.gerber[1];
				dftROI.n = defectList[n].n;
				dftROI.type = DefectType::Heterochrome;
				dftROI.layer = abstract;
				dftROI.isCuda = false;
				gOutputInfo->defect.emplace_back(dftROI);

				numDefectCommon++;
			}
		}
	}

	log->info("普通算法检测到的缺陷数量："+std::to_string(numDefectCommon));

	int numDefectCUDA = 0;
	for (int n=0;n<defectListCuda.size();n++)
	{
		for (int i = 0; i < defectListCuda[n].dft.size(); i++)
		{
			
			DefectRoi dftROI;
			dftROI.image[0] = defectListCuda[n].dft[i].roi.x;
			dftROI.image[1] = defectListCuda[n].dft[i].roi.y;
			dftROI.image[2] = defectListCuda[n].dft[i].roi.width;
			dftROI.image[3] = defectListCuda[n].dft[i].roi.height;
			dftROI.image[4] = defectListCuda[n].dft[i].nPcs;
			img2gerb(dftROI.image[1], dftROI.image[0],
				imageIndex, isTop,
				dftROI.gerber[1], dftROI.gerber[0]);
			img2gerb(dftROI.image[1] + dftROI.image[3],
				dftROI.image[0] + dftROI.image[2],
				imageIndex, isTop,
				dftROI.gerber[3], dftROI.gerber[2]);
			dftROI.gerber[2] -= dftROI.gerber[0];
			dftROI.gerber[3] -= dftROI.gerber[1];
			dftROI.n = defectListCuda[n].dft[i].nImg;
			dftROI.type = DefectType::Heterochrome;
			dftROI.layer = defectListCuda[n].dft[i].abstract;
			dftROI.isCuda = true;
			gOutputInfo->defect.emplace_back(dftROI);

			numDefectCUDA++;
		}
	}
	log->info("CUDA算法检测到的缺陷数量：" + std::to_string(numDefectCUDA));
	log->info("检测到的缺陷数量（累积值）：" + std::to_string(gOutputInfo->defect.size()));

#ifdef TEST
	int ndef = 0;
	cv::Mat _show;
    cv::cvtColor(input, _show, cv::COLOR_RGB2BGR);
	for (int n = 0; n < defectList.size(); n++)
	{
		for (int i = 0; i < defectList[n].roi.size(); i++)
		{
			Layer layer = defectList[n].roi[i].abstract;
			for (int j = 0; j < defectList[n].roi[i].info.size(); j++)
			{
				cv::Rect roi = { defectList[n].roi[i].info[j].rect.x, defectList[n].roi[i].info[j].rect.y, defectList[n].roi[i].info[j].rect.width, defectList[n].roi[i].info[j].rect.height };
				cv::Point _center = { roi.x + roi.width / 2, roi.y + roi.height / 2 };
				//cv::rectangle(input, roi, cv::Scalar(0, 0, 255), 1);
				cv::ellipse(_show, _center, cv::Size(roi.width + 3, roi.height + 3), 0, 0, 360, cv::Scalar(0, 0, 255), 2);
				cv::putText(_show, std::to_string(n) + std::to_string(layer) + std::to_string(j), _center, cv::HersheyFonts::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0, 255, 255), 1);
				//cv::rectangle(input, cv::Point(roi.x, roi.y), cv::Point(roi.x + roi.width - 1, roi.y + roi.height - 1), cv::Scalar(0, 0, 255), 2);
				ndef++;
			}

		}

		for (int i = 0; i < defectListCuda[n].dft.size(); i++)
		{
			cv::Rect roi = { defectListCuda[n].dft[i].roi.x, defectListCuda[n].dft[i].roi.y, defectListCuda[n].dft[i].roi.width, defectListCuda[n].dft[i].roi.height };
			cv::Point _center = { roi.x + roi.width / 2, roi.y + roi.height / 2 };
			//cv::rectangle(input, roi, cv::Scalar(0, 0, 255), 1);
			cv::ellipse(_show, _center, cv::Size(roi.width, roi.height), 0, 0, 360, cv::Scalar(0, 255, 0), 2);
			cv::putText(_show, std::to_string(n) + std::to_string(defectListCuda[n].dft[i].abstract) + std::to_string(i), _center, cv::HersheyFonts::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0, 255, 255), 1);
			//cv::rectangle(input, cv::Point(roi.x, roi.y), cv::Point(roi.x + roi.width - 1, roi.y + roi.height - 1), cv::Scalar(0, 0, 255), 2);
			ndef++;
		}
	}

	static int ntest = 0;
	//cv::cvtColor(_show, _show, cv::ColorConversionCodes::COLOR_BGR2RGB);
	cv::rectangle(_show, cv::Rect(cv::Point(30, 30), cv::Size(5000, 200)), cv::Scalar(255, 255, 255), -1);
	cv::putText(_show, "Defects number:" + std::to_string(ndef), cv::Point(30, 180), 0, 5, cv::Scalar(0, 0, 255), 5, 8);
	cv::imwrite(std::to_string(ntest++) + "_f.jpg", _show);
#endif
	
	/*static int nImgLocation = 0;
	cv::Mat _imgLocation = input.clone();
	location(_imgLocation, imageIndex, 1, 1);
	cv::cvtColor(_imgLocation, _imgLocation, cv::ColorConversionCodes::COLOR_BGR2RGB);
	cv::imwrite(std::to_string(nImgLocation++) + "__location.jpg", _imgLocation);
*/
	////static int nImgItem = 0;
	//unsigned char *ptrImgItem = (unsigned char *)malloc(input.cols*input.rows * sizeof(unsigned char) * 3);
	//cv::Mat imgItem(input.rows, input.cols, CV_8UC3, ptrImgItem);
	//viewItems(imgItem, gParam,imageIndex, 1, 1);
	//cv::imwrite(std::to_string(nImgItem++) + "_viewItems.jpg", imgItem);
	//free(ptrImgItem);

	/*static int nImgVisual = 0;
	unsigned char *ptrImgVisual = (unsigned char *)malloc(input.cols*input.rows * sizeof(unsigned char) * 3);
	cv::Mat imgVisual(input.rows, input.cols, CV_8UC3, ptrImgVisual);
	Inspector::visualItems(imgVisual, gParam, imageIndex);
	cv::cvtColor(imgVisual, imgVisual, cv::ColorConversionCodes::COLOR_BGR2RGB);
	cv::imwrite(std::to_string(nImgVisual++) + "_visualItems.jpg", imgVisual);
	free(ptrImgVisual);*/

#ifdef _TIME_LOG_
	std::chrono::steady_clock::time_point time4 = std::chrono::steady_clock::now();

	timeCastSingle = std::chrono::duration_cast<std::chrono::milliseconds>(time4 - time3);
	timeCastFull = std::chrono::duration_cast<std::chrono::milliseconds>(time4 - timeStart);
	time4CastFull += timeCastSingle.count();
	//std::ofstream file(log_time_name, std::ios::out | std::ios::app);
	file <<"  "<< std::left << std::setw(32)<< strFunction + "("+std::to_string(imageIndex + 1) + "/" + std::to_string(gImageNumber)+")4 数据打包"
		<< std::right << std::setw(16) << timeCastFull.count()
		<< std::right << std::setw(16) << timeCastSingle.count() 
		<< std::right << std::setw(16) << time4CastFull << std::endl;

#endif


	if (imageIndex == (gImageNumber - 1))
	{
#ifdef _TIME_LOG_
		
#endif
		log->info("********5/5 缺陷图像整理 " +std::to_string(imageIndex + 1) + " / " + std::to_string(gImageNumber) + "*******");
		cv::Mat result;
		try
		{
			log->info("缺陷图像开始整理");

			imageFusion(gOutputInfo->scale, result, isTop);
			//32bit aligning
			int width = BYTEALIGNING(result.cols);
			int height = result.rows;
			int step = width * 3;
			cv::resize(result, result, cv::Size(width, height));
			gOutputInfo->image.width = width;
			gOutputInfo->image.height = height;
			gOutputInfo->image.step = width * 3;
			//cv::cvtColor(result, result, cv::COLOR_RGB2BGR);
			memcpy(gOutputInfo->image.ptr, result.data, gOutputInfo->image.step * gOutputInfo->image.height);
			
			log->info("缺陷图像整理完毕");

		}
		catch (...)
		{
			log->error("缺陷图像整理出现错误！");

			return -1;
		}
	}

#ifdef _TIME_LOG_
	std::chrono::steady_clock::time_point time5 = std::chrono::steady_clock::now();
	timeCastSingle = std::chrono::duration_cast<std::chrono::milliseconds>(time5 - time4);
	timeCastFull = std::chrono::duration_cast<std::chrono::milliseconds>(time5 - timeStart);
	time5CastFull += timeCastSingle.count();
	//std::ofstream file(log_time_name, std::ios::out | std::ios::app);
	file <<"  "<<std::left << std::setw(32) <<strFunction + "("+std::to_string(imageIndex + 1) + "/" + std::to_string(gImageNumber)  +")5 数据整理"
		<< std::right << std::setw(16) << timeCastFull.count()
		<< std::right << std::setw(16) << timeCastSingle.count() 
		<< std::right << std::setw(16) << time5CastFull <<std::endl;
	file.close();

	if (imageIndex == (gImageNumber - 1))
	{
		time1CastFull = 0;
		time2CastFull = 0;
		time3CastFull = 0;
		time4CastFull = 0;
		time5CastFull = 0;
	}

#endif

	log->info("========缺陷检测算法完毕 (" + std::to_string(imageIndex + 1) + " / " + std::to_string(gImageNumber) + ")========");
	return 0;
}

int Inspector::processInterface(const cv::Mat & input, int imageIndex, int isTop, std::vector<DefectRoi>& defects, bool isChange)
{
    iniLogAlg();

    auto log = spdlog::get("log");

    if (isTop && gTrainParamTop.empty())
    {
        log->error("未训练正面参数，或正面参数训练有误！");
        return -1;
    }

    if (!isTop && gTrainParamBack.empty())
    {
        log->error("未训练反面参数，或反面参数训练有误！");
        return -1;
    }

    TrainParam *pTrainParam;
    ConfigParam *pConfigParam;

    if (isTop)
    {
        pTrainParam = &gTrainParamTop;
        pConfigParam = &gParamTop;
    }
    else
    {
        pTrainParam = &gTrainParamBack;
        pConfigParam = &gParamBack;
    }

    int err = 0;

    cv::Mat imgSegement;

    cv::cvtColor(input, imgSegement, cv::ColorConversionCodes::COLOR_BGR2RGB);

    if(isChange == true)
    {
        for (int i = 0; i < gImageNumber; i++)
        {
            gItems[i].clear();
        }
    }

    try
    {
        if (gItems[imageIndex].size() <= 0)
        {
            err = doSegment(imgSegement, imageIndex, isTop, gItems[imageIndex]);
        }

        if (err != 0)
        {
            log->error("分割失败!");
            return -3;
        }
    }
    catch (...)
    {
        log->error("分割失败!");
        return -3;
    }

    std::vector<std::future<int>> vecFunc;

    int nPcs = gItems[imageIndex].size();

    std::vector<DefectInfo> defectList(nPcs);

    log->info("Commom Defects Inspect Alg Start.");

    for (int n = 0; n < nPcs; n++)
    {
        defectList[n].n = imageIndex;

        defectList[n].nPcs = n;

        log->info("Mulit Thread Mode.");

        log->info("PCS " + std::to_string(n + 1) + "/" + std::to_string(nPcs) + " Start.");

        vecFunc.emplace_back(threadPool->submit(
                            AlgBase::itemInspectorV2,
                            std::ref(input),
                            std::ref(isTop),
                            std::ref(imageIndex),
                            std::ref(gItems[imageIndex][n]),
                            std::ref(*pTrainParam),
                            std::ref(*pConfigParam),
                            std::ref(defectList[n])));

        log->info("PCS " + std::to_string(n + 1) + "/" + std::to_string(nPcs) + " End.");
    }

    for (int v = 0; v < vecFunc.size(); v++)
    {
        vecFunc[v].get();
    }
    
    int numDefectCommon = 0;

    defects.clear();

    defectsForInterface.clear();

    for (int n = 0; n < defectList.size(); n++)
    {
        for (int i = 0; i < defectList[n].roi.size(); i++)
        {
            Layer abstract = defectList[n].roi[i].abstract;

            for (int j = 0; j<defectList[n].roi[i].info.size(); j++)
            {
                DefectRoi dftROI;

                dftROI.image[0] = defectList[n].roi[i].info[j].rect.x;

                dftROI.image[1] = defectList[n].roi[i].info[j].rect.y;

                dftROI.image[2] = defectList[n].roi[i].info[j].rect.width;

                dftROI.image[3] = defectList[n].roi[i].info[j].rect.height;

                dftROI.image[4] = defectList[n].nPcs;

                dftROI.itemIdx = defectList[n].roi[i].info[j].id;

                dftROI.isTop = isTop;

                img2gerb(dftROI.image[1], dftROI.image[0],
                    imageIndex, isTop,
                    dftROI.gerber[1], dftROI.gerber[0]);

                img2gerb(dftROI.image[1] + dftROI.image[3],
                    dftROI.image[0] + dftROI.image[2],
                    imageIndex, isTop,
                    dftROI.gerber[3], dftROI.gerber[2]);

                dftROI.gerber[2] -= dftROI.gerber[0];

                dftROI.gerber[3] -= dftROI.gerber[1];

                dftROI.n = defectList[n].n;

                dftROI.type = DefectType::Heterochrome;

                dftROI.layer = abstract;

                dftROI.isCuda = false;

                defects.push_back(dftROI);

                numDefectCommon++;
            }
        }
    }

    defectsForInterface.assign(defects.begin(), defects.end());

    log->info("普通算法检测到的缺陷数量：" + std::to_string(numDefectCommon));

    return 0;
}

//GUI
int Inspector::process(cv::Mat &input, int imageIndex, int isTop, ConfigParam & configParam, std::vector<DefectRoi>& defects)
{
	if (isTop && !bPosParam)
	{
		std::cout << "未训练正面参数，或正面参数训练有误" << std::endl;
		return -1;
	}
	if (!isTop && !bNegParam)
	{
		std::cout << "未训练反面参数，或反面参数训练有误" << std::endl;
		return -1;
	}

	

	int err = 0;
	cv::Mat imgSegement;
	cv::cvtColor(input, imgSegement, cv::ColorConversionCodes::COLOR_BGR2RGB);


	try
	{
#ifndef OPENPREPROCESS
        input.copyTo(gImageSpace[imageIndex]);
#else
        //cv::LUT(input, lut, gImageSpace[imageIndex]);
        //epmFilter(learnImage, gImageSpace[imageIndex], 5, 0.01);
#endif
		err = doSegment(imgSegement, imageIndex, isTop, gItems[imageIndex]);
		if (err != 0)
		{
			printf("分割失败!\n");
			return -3;
		}
	}
	catch (...)
	{
		printf("分割失败!\n");
		return -3;
	}

	std::vector<std::future<int>> vecFunc;

	//cv::Mat blurMat;
	//cv::medianBlur(input, blurMat, 3);
	//cv::split(gImageSpace[imageIndex], channels);

	int nPcs = gItems[imageIndex].size();
	std::vector<DefectInfo> defectList(nPcs);
	std::vector<DefectCudaInfo> defectListCuda(nPcs);
    //input.copyTo(gImageSpace[imageIndex]);

#ifndef _DEBUG

#if USEING_DL

	vecFunc.emplace_back(threadPool->submit(
		AlgBase::itemInspectorCuda,
		&input,
		std::ref(gItems[imageIndex]),
		std::ref(configParam),
		std::ref(defectListCuda)
	));

#endif
   
#endif

	for (int n = 0; n < nPcs; n++)
	{
		defectList[n].n = imageIndex;
		defectList[n].nPcs = n;

		//AlgBase::itemInspector(
		//	&input,
		//	isTop,
		//	gItems[imageIndex][n],
		//	configParam,
		//	defectList[n],
		//	true);

		vecFunc.emplace_back(threadPool->submit(
			AlgBase::itemInspector,
            &input,
			isTop,
			std::ref(gItems[imageIndex][n]),
			std::ref(configParam),
			std::ref(defectList[n]),
            true)
		);
	}

	for (int v = 0; v < vecFunc.size(); v++)
	{
		vecFunc[v].get();
	}

    defects.clear();
    gImageSpace[imageIndex] = input.clone();
    cv::Mat _draw = input.clone();
	for (int n = 0; n < defectList.size(); n++)
	{
		//for (int i = 0; i < defectList[n].roi.size(); i++)
		//{
  //          cv::Rect roi = { defectList[n].roi[i].x, defectList[n].roi[i].y, defectList[n].roi[i].width, defectList[n].roi[i].height };
  //          cv::Point _center = { roi.x + roi.width / 2, roi.y + roi.height / 2 };
  //          //cv::rectangle(input, roi, cv::Scalar(0, 0, 255), 1);
  //          cv::ellipse(_draw, _center, cv::Size(roi.width + 3, roi.height + 3), 0, 0, 360, cv::Scalar(0, 0, 255), 2);

		//	defects.push_back({ { defectList[n].roi[i].x, defectList[n].roi[i].y, defectList[n].roi[i].width, defectList[n].roi[i].height, defectList[n].nPcs },
		//						{ 0, 0, 0, 0, 0 },
		//						DefectType::Heterochrome,
		//						imageIndex ,
  //                              false});
		//}
  //      
  //      for (int i = 0; i < defectListCuda[n].roi.size(); i++)
  //      {
  //          defects.push_back({ { defectListCuda[n].roi[i].x, defectListCuda[n].roi[i].y, defectListCuda[n].roi[i].width, defectListCuda[n].roi[i].height, defectListCuda[n].nPcs },
  //                          { 0, 0, 0, 0, 0 },
  //                              DefectType::Heterochrome,
  //                              imageIndex ,
  //                              true });
  //      }


		for (int j = 0; j < defectList[n].roi.size(); j++)
		{
			Layer layer = defectList[n].roi[j].abstract;
			for (int k = 0; k < defectList[n].roi[j].info.size(); k++)
			{
				cv::Rect roi = { defectList[n].roi[j].info[k].rect.x, defectList[n].roi[j].info[k].rect.y, defectList[n].roi[j].info[k].rect.width, defectList[n].roi[j].info[k].rect.height };
				cv::Point _center = { roi.x + roi.width / 2, roi.y + roi.height / 2 };
				//cv::rectangle(input, roi, cv::Scalar(0, 0, 255), 1);
				cv::ellipse(_draw, _center, cv::Size(roi.width + 3, roi.height + 3), 0, 0, 360, cv::Scalar(0, 0, 255), 2);
				//cv::putText(_draw, std::to_string(n) + std::to_string(layer) + std::to_string(j), _center, cv::HersheyFonts::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0, 255, 255), 1);
				//cv::rectangle(input, cv::Point(roi.x, roi.y), cv::Point(roi.x + roi.width - 1, roi.y + roi.height - 1), cv::Scalar(0, 0, 255), 2);


				defects.push_back({ { defectList[n].roi[j].info[k].rect.x, defectList[n].roi[j].info[k].rect.y, defectList[n].roi[j].info[k].rect.width, defectList[n].roi[j].info[k].rect.height, defectList[n].nPcs },
				{ 0, 0, 0, 0, 0 },
					DefectType::Heterochrome,
					layer,
					imageIndex,
					false });

			}

		}

		for (int j = 0; j < defectListCuda[n].dft.size(); j++)
		{
			cv::Rect roi = { defectListCuda[n].dft[j].roi.x, defectListCuda[n].dft[j].roi.y, defectListCuda[n].dft[j].roi.width, defectListCuda[n].dft[j].roi.height };
			cv::Point _center = { roi.x + roi.width / 2, roi.y + roi.height / 2 };
			//cv::rectangle(input, roi, cv::Scalar(0, 0, 255), 1);
			cv::ellipse(_draw, _center, cv::Size(roi.width, roi.height), 0, 0, 360, cv::Scalar(0, 255, 0), 2);
			cv::putText(_draw,std::to_string(defectListCuda[n].dft[j].abstract),_center,0,0.3,cv::Scalar(0,255,0));
			defects.push_back({ { defectListCuda[n].dft[j].roi.x, defectListCuda[n].dft[j].roi.y, defectListCuda[n].dft[j].roi.width, defectListCuda[n].dft[j].roi.height, defectListCuda[n].dft[j].nPcs },
			{ 0, 0, 0, 0, 0 },
				DefectType::Heterochrome,
				defectListCuda[n].dft[j].abstract,
				imageIndex,
				true });
			//cv::rectangle(input, cv::Point(roi.x, roi.y), cv::Point(roi.x + roi.width - 1, roi.y + roi.height - 1), cv::Scalar(0, 0, 255), 2);
		}
	}

	return 0;
}

//实时显示
int Inspector::processRealtime(ConfigParam & configParam, int isTop, std::vector<DefectRoi>& defects)
{
	if (isTop && !bPosParam)
	{
		std::cout << "未训练正面参数，或正面参数训练有误" << std::endl;
		return -1;
	}
	if (!isTop && !bNegParam)
	{
		std::cout << "未训练反面参数，或反面参数训练有误" << std::endl;
		return -1;
	}

	std::vector<std::future<int>> vecFunc;

	for (int i = 0; i < gImageNumber; i++)
	{

		int nPcs = gItems[i].size();

		std::vector<DefectInfo> defectList(nPcs);
		std::vector<DefectCudaInfo> defectListCuda(nPcs);


#ifndef _DEBUG
#if USEING_DL

		vecFunc.emplace_back(threadPool->submit(
			AlgBase::itemInspectorCuda,
			&gImageSpace[i],
			std::ref(gItems[i]),
			std::ref(configParam),
			std::ref(defectListCuda)
		));

#endif
        
#endif

		for (int n = 0; n < nPcs; n++)
		{

			defectList[n].n = i;
			defectList[n].nPcs = n;

			//AlgBase::itemInspector(&gImageSpace[i], isTop, gItems[i][n], configParam, defectList[n]);
			vecFunc.emplace_back(threadPool->submit(
				AlgBase::itemInspector,
                &gImageSpace[i],
				isTop,
				std::ref(gItems[i][n]),
				std::ref(configParam),
				std::ref(defectList[n]),
                true)
			);
		}
		
		for (int v = 0; v < vecFunc.size(); v++)
		{
			vecFunc[v].get();
		}

        vecFunc.clear();

        cv::Mat _draw = gImageSpace[i].clone();
		for (int n = 0; n < nPcs; n++)
		{
            //for (int j = 0; j < defectList[n].roi.size(); j++)
            //{
            //    cv::Rect roi = { defectList[n].roi[j].x, defectList[n].roi[j].y, defectList[n].roi[j].width, defectList[n].roi[j].height };
            //    cv::Point _center = { roi.x + roi.width / 2, roi.y + roi.height / 2 };
            //    //cv::rectangle(input, roi, cv::Scalar(0, 0, 255), 1);
            //    cv::ellipse(_draw, _center, cv::Size(roi.width + 3, roi.height + 3), 0, 0, 360, cv::Scalar(0, 0, 255), 2);

            //    defects.push_back({ { defectList[n].roi[j].x, defectList[n].roi[j].y, defectList[n].roi[j].width, defectList[n].roi[j].height, defectList[n].nPcs },
            //                        { 0, 0, 0, 0, 0 },
            //                            DefectType::Heterochrome,
            //                            i,
            //                            false });
            //}

            //for (int j = 0; j < defectListCuda[n].roi.size(); j++)
            //{
            //    defects.push_back({ { defectListCuda[n].roi[j].x, defectListCuda[n].roi[j].y, defectListCuda[n].roi[j].width, defectListCuda[n].roi[j].height, defectListCuda[n].nPcs },
            //                        { 0, 0, 0, 0, 0 },
            //                            DefectType::Heterochrome,
            //                            i,
            //                            true });
            //}

			for (int j = 0; j < defectList[n].roi.size(); j++)
			{
				Layer layer = defectList[n].roi[j].abstract;
				for (int k = 0; k < defectList[n].roi[j].info.size(); k++)
				{
					cv::Rect roi = { defectList[n].roi[j].info[k].rect.x, defectList[n].roi[j].info[k].rect.y, defectList[n].roi[j].info[k].rect.width, defectList[n].roi[j].info[k].rect.height };
					cv::Point _center = { roi.x + roi.width / 2, roi.y + roi.height / 2 };
					//cv::rectangle(input, roi, cv::Scalar(0, 0, 255), 1);
					cv::ellipse(_draw, _center, cv::Size(roi.width + 3, roi.height + 3), 0, 0, 360, cv::Scalar(0, 0, 255), 2);
					//cv::putText(_draw, std::to_string(n) + std::to_string(layer) + std::to_string(j), _center, cv::HersheyFonts::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0, 255, 255), 1);
					//cv::rectangle(input, cv::Point(roi.x, roi.y), cv::Point(roi.x + roi.width - 1, roi.y + roi.height - 1), cv::Scalar(0, 0, 255), 2);


					defects.push_back({ { defectList[n].roi[j].info[k].rect.x, defectList[n].roi[j].info[k].rect.y, defectList[n].roi[j].info[k].rect.width, defectList[n].roi[j].info[k].rect.height, defectList[n].nPcs },
						                        { 0, 0, 0, 0, 0 },
						                            DefectType::Heterochrome,
													layer,
						                            i,
						                            false });
					
				}

			}

			for (int j = 0; j < defectListCuda[n].dft.size(); j++)
			{
				cv::Rect roi = { defectListCuda[n].dft[j].roi.x, defectListCuda[n].dft[j].roi.y, defectListCuda[n].dft[j].roi.width, defectListCuda[n].dft[j].roi.height };
				cv::Point _center = { roi.x + roi.width / 2, roi.y + roi.height / 2 };
				//cv::rectangle(input, roi, cv::Scalar(0, 0, 255), 1);
				cv::ellipse(_draw, _center, cv::Size(roi.width, roi.height), 0, 0, 360, cv::Scalar(0, 255, 0), 2);

				defects.push_back({ { defectListCuda[n].dft[j].roi.x, defectListCuda[n].dft[j].roi.y, defectListCuda[n].dft[j].roi.width, defectListCuda[n].dft[j].roi.height, defectListCuda[n].dft[j].nPcs },
											{ 0, 0, 0, 0, 0 },
												DefectType::Heterochrome,
												defectListCuda[n].dft[j].abstract,
												i,
												true });
				//cv::rectangle(input, cv::Point(roi.x, roi.y), cv::Point(roi.x + roi.width - 1, roi.y + roi.height - 1), cv::Scalar(0, 0, 255), 2);
			}
		}
        std::cout<<defects.size()<<std::endl;
	}
	return 0;
}

//?-?ú?¤??àaμ?·?ê???DD?¨??D§1?2é?′￡?
int Inspector::location(
	cv::Mat &output, 
	int imageNumber, 
	float scale, 
	int brushWidth)
{
	iniLogAlg();
	auto log = spdlog::get("log");

	std::string strFunction(__FUNCTION__);
	log->info("========" + strFunction + "========");

	if (gItems[imageNumber].size() <= 0)
	{

		return -1;
	}
	//cv::Mat rgb[3];
	//cv::Mat _output;

	//cv::Size reSize = { (int)(gImageSpace[imageNumber].cols * scale), (int)(gImageSpace[imageNumber].rows * scale) };
	//output = cv::Mat(reSize, CV_8UC3);
	cv::resize(gImageSpace[imageNumber], output, output.size());
	float scaleX = 1.f * output.cols / gImageSpace[imageNumber].cols;
	float scaleY = 1.f * output.rows / gImageSpace[imageNumber].rows;
	cv::Mat mask, reMask, maskCutted;

	int x = 0, y = 0, width = 0, height = 0, ex = 0, ey = 0, sx = 0, sy = 0;

	cv::Rect maskRoi;
	std::vector<std::vector<cv::Point>> vecContours;
	//cv::Mat _temp;
	//cv::Rect rect;

	int index = 0;
	for (int i = 0; i < gItems[imageNumber].size(); i++)
	{
		int indexPcs = gItems[imageNumber][i].iID;
		for (int j = 0; j < gItems[imageNumber][i].itemsRegions.size(); j++)
		{
			Layer lay = gItems[imageNumber][i].itemsRegions[j].type;


			//if ( /*lay != pcsContourLay
			//	 &&*/ /*lay != drillLay_unposition
			//	|| */lay != lineLay_pad)
			//{
			//	continue;
			//}

			for (int k = 0; k < gItems[imageNumber][i].itemsRegions[j].items.size(); k++)
			{
				mask = gItems[imageNumber][i].itemsRegions[j].items[k].mask;

				x = gItems[imageNumber][i].itemsRegions[j].items[k].iOffsetX * scaleX;
				y = gItems[imageNumber][i].itemsRegions[j].items[k].iOffsetY * scaleY;
				width = mask.cols * scaleX;
				height = mask.rows * scaleY;
				if (width <= 0 || height <= 0)	continue;
				cv::resize(mask, reMask, cv::Size(width, height), 0, 0, 0);

				sx = __max(0, x);
				sy = __max(0, y);
				ex = __min(x + width, output.cols);
				ey = __min(y + height, output.rows);
				width = ex - sx;
				height = ey - sy;

				maskRoi = { sx - x,0,width,height };

				if (ex <= 0 || width <= 0)
					continue;

				//extract edge
				maskCutted = reMask(maskRoi);
				//cv::imwrite(std::to_string(index) + "-" + std::to_string(indexPcs)+".jpg", maskCutted);
				int index = gItems[imageNumber][i].itemsRegions[j].items[k].iID;
				cv::findContours(maskCutted, vecContours, cv::RetrievalModes::RETR_LIST, cv::ContourApproximationModes::CHAIN_APPROX_NONE);
				for (int l = 0; l < vecContours.size(); l++)
				{
					cv::drawContours(output, vecContours, l, cv::Scalar(0, 255, 0), 1, 8, cv::noArray(), INT_MAX, cv::Point(sx, sy));

					cv::Rect rect = cv::boundingRect(vecContours[l]);
					cv::putText(output, std::to_string(index) + "-" + std::to_string(indexPcs), cv::Point(rect.tl() / 2 + rect.br() / 2) + cv::Point(sx, sy), 0, 0.5, cv::Scalar(255, 255, 255));
				}
			}
		}
	}


	log->info("========?¨??íê±?========");
	return 0;
}
int Inspector::viewItems(cv::Mat & output, const ConfigParam &configParam, int imageNumber, float scale, int brushWidth)
{
	iniLogAlg();
	auto log = spdlog::get("log");

	std::string strFunction(__FUNCTION__);
	log->info("========" + strFunction + "========");

	if (gItems[imageNumber].size() <= 0)
	{

		return -1;
	}
	//cv::Mat rgb[3];
	//cv::Mat _output;

	//cv::Size reSize = { (int)(gImageSpace[imageNumber].cols * scale), (int)(gImageSpace[imageNumber].rows * scale) };
	//output = cv::Mat(reSize, CV_8UC3);
	cv::resize(gImageSpace[imageNumber], output, output.size());
	float scaleX = 1.f * output.cols / gImageSpace[imageNumber].cols;
	float scaleY = 1.f * output.rows / gImageSpace[imageNumber].rows;
	cv::Mat mask, reMask, maskCutted;

	cv::Mat output1 = output.clone();
	output = 0;
	int x = 0, y = 0, width = 0, height = 0, ex = 0, ey = 0, sx = 0, sy = 0;

	cv::Rect maskRoi;
	std::vector<std::vector<cv::Point>> vecContours;
	//cv::Mat _temp;
	//cv::Rect rect;

	int index = 0;
	for (int i = 0; i < gItems[imageNumber].size(); i++)
	{
		int indexPcs = gItems[imageNumber][i].iID;
		for (int j = 0; j < gItems[imageNumber][i].itemsRegions.size(); j++)
		{
			Layer lay = gItems[imageNumber][i].itemsRegions[j].type;


			if ( lay == pcsContourLay)
			{
				continue;
			}

			int id = gItems[imageNumber][i].itemsRegions[j].iID;
			for (int k = 0; k < gItems[imageNumber][i].itemsRegions[j].items.size(); k++)
			{
				mask = gItems[imageNumber][i].itemsRegions[j].items[k].mask;

				
				width = mask.cols * scaleX;
				height = mask.rows * scaleY;
				if (width <= 0 || height <= 0)	continue;
				cv::resize(mask, reMask, cv::Size(width, height), 0, 0, 0);

				x = gItems[imageNumber][i].itemsRegions[j].items[k].iOffsetX * scaleX;
				y = gItems[imageNumber][i].itemsRegions[j].items[k].iOffsetY * scaleY;
				sx = __max(0, x);
				sy = __max(0, y);
				ex = __min(x + width, output.cols);
				ey = __min(y + height, output.rows);
				width = ex - sx;
				height = ey - sy;

				maskRoi = { sx - x,0,width,height };

				if (ex <= 0 || width <= 0)
					continue;

				//extract edge
				maskCutted = reMask(maskRoi);
				//cv::imwrite(std::to_string(index) + "-" + std::to_string(indexPcs)+".jpg", maskCutted);
				//int index = gItems[imageNumber][i].itemsRegions[j].items[k].iID;
				
				for (int r = 0; r < maskCutted.rows; r++)
				{
					unsigned char *pMask = maskCutted.ptr<unsigned char>(r);

					for (int c = 0; c < maskCutted.cols; c++)
					{
						if (pMask[c] == 255)
						{
							cv::Vec3b color=output.at<cv::Vec3b>(y + r, x + maskRoi.tl().x + c);
							if (color[0]==0&& color[1] == 0&& color[2] == 0)
							{
								output.at<cv::Vec3b>(y + r, x + maskRoi.tl().x + c) = cv::Vec3b((id + 1) * 20, (id + 1) * 20, (id + 1) * 20);
							}
							else
							{
								output.at<cv::Vec3b>(y + r, x + maskRoi.tl().x + c) = cv::Vec3b(0, 255, 255);
							}
							//output.at<cv::Vec3b>(y + r, x + maskRoi.tl().x + c) = cv::Vec3b((id + 1) * 20,(id + 1) * 20 ,(id + 1) * 20 );
						}

					}

				}

				if (lay != lineLay_pad)
				{
					continue;
				}
				cv::Mat kernel = cv::getStructuringElement(cv::MorphShapes::MORPH_ELLIPSE, cv::Size(13, 13));
				cv::Mat imgMor;
				cv::morphologyEx(maskCutted, imgMor, cv::MorphTypes::MORPH_ERODE,
					kernel,
					cv::Point(-1, -1),
					1,
					cv::BorderTypes::BORDER_CONSTANT,
					0);

				for (int r = 0; r < imgMor.rows; r++)
				{
					unsigned char *pMask = imgMor.ptr<unsigned char>(r);

					for (int c = 0; c < imgMor.cols; c++)
					{
						if (pMask[c] == 255)
						{
							output1.at<cv::Vec3b>(y + r, x + maskRoi.tl().x + c) = cv::Vec3b(255, 255, 255);
							//output.at<cv::Vec3b>(y + r, x + maskRoi.tl().x + c) = cv::Vec3b((id + 1) * 20,(id + 1) * 20 ,(id + 1) * 20 );
						}

					}

				}


			}
		}
	}

	static int num = 0;
	cv::cvtColor(output1, output1, cv::ColorConversionCodes::COLOR_BGR2RGB);
	cv::imwrite(std::to_string(num++)+"_erode.jpg",output1);

	log->info("========????íê±?========");
	return 0;
}

int Inspector::visualItems(cv::Mat & output, const ConfigParam &configParam, const int imageNumber)
{
	iniLogAlg();
	auto log = spdlog::get("log");

	std::string strFunction(__FUNCTION__);
	log->info("========" + strFunction + "========");

	if (gItems[imageNumber].size() <= 0)
	{
		return -1;
	}
	if (gImageSpace[imageNumber].rows!=output.rows
		|| gImageSpace[imageNumber].cols != output.cols)
	{
		return -1;
	}

	output = gImageSpace[imageNumber].clone();
	cv::Mat mask, reMask, maskCutted;
	int x = 0, y = 0, width = 0, height = 0, ex = 0, ey = 0, sx = 0, sy = 0;

	cv::Rect maskRoi;
	std::vector<std::vector<cv::Point>> vecContours;
	//cv::Mat _temp;
	//cv::Rect rect;

	int index = 0;
	for (int i = 0; i < gItems[imageNumber].size(); i++)
	{
		int indexPcs = gItems[imageNumber][i].iID;
		for (int j = 0; j < gItems[imageNumber][i].itemsRegions.size(); j++)
		{
			Layer lay = gItems[imageNumber][i].itemsRegions[j].type;

			int id =  gItems[imageNumber][i].itemsRegions[j].iID;

			
			int a = id / 9;
			int b = (id % 9) / 3;
			int c = ((id % 9) % 3) % 3;
			
			bool isInspect = false;
			cv::Vec3b color = cv::Vec3b(b * 127.5, a * 127.5, c * 127.5);
			int kernelSize = 0;
			switch (lay)
			{
			case NoType:
				break;
			case lineLay_org:
				break;
			case lineLay_pad:
				if (configParam.PadParam.vaild)
				{
					kernelSize = configParam.PadParam.shrinkSize * 2 + 1;
					isInspect = true;
				}
				break;
			case lineLay_nest:
				if (configParam.TransprencyParam.vaild)
				{
					kernelSize = configParam.TransprencyParam.shrinkSize * 2 + 1;
					isInspect = true;
				}
				break;
			case lineLay_conduct:
				break;
			case lineLay_finger:
				break;
			case lineLay_base:
				if (configParam.TransprencyParam.vaild)
				{
					kernelSize = configParam.TransprencyParam.shrinkSize * 2 + 1;
					isInspect = true;
				}
				break;
			case lineLay_baojiao:
				break;
			case lineLay_goldRegion:
				break;
			case lineLay_ungoldCopper:
				break;
			case lineLay_goldCopper:
				break;
			case lineBackLay_org:
				break;
			case lineBackLay_base:
				break;
			case charLay_varMask:
				break;
			case charLay_fix:
				if (configParam.CharParam.vaild)
				{
					kernelSize = configParam.CharParam.shrinkSize * 2 + 1;
					isInspect = true;
				}
				break;
			case stiffenerLay_steel:
				if (configParam.SteelParam.vaild)
				{
					kernelSize = configParam.SteelParam.shrinkSize * 2 + 1;
					isInspect = true;
				}
				break;
			case printLay_EMI:
				if (configParam.OpacityParam.vaild)
				{
					kernelSize = configParam.OpacityParam.shrinkSize * 2 + 1;
					isInspect = true;
				}
				break;
			case printLay_lvyou:
				break;
			case drillLay_org:
				break;
			case drillLay_position:
				break;
			case drillLay_unposition:
				break;
			case pcsContourLay:
				break;
			case pcsMarkLay:
				break;
			case carveLay:
				if (configParam.CarveParam.vaild)
				{
					kernelSize = configParam.CarveParam.shrinkSize * 2 + 1;
					isInspect = true;
				}
				break;
			case holingThroughLay:
				if (configParam.HoleParam.vaild)
				{
					kernelSize = configParam.HoleParam.shrinkSize * 2 + 1;
					isInspect = true;
				}
				break;
			case uselessLay:
				break;
			case exemptionLay:
				break;
			case NLayer:
				break;
			default:
				break;
			}

			if (!isInspect)
			{
				continue;
			}

			cv::Mat kernel = cv::Mat::ones(cv::Size(kernelSize, kernelSize), CV_8UC1);

			for (int k = 0; k < gItems[imageNumber][i].itemsRegions[j].items.size(); k++)
			{
				mask = gItems[imageNumber][i].itemsRegions[j].items[k].mask;

				width = mask.cols ;
				height = mask.rows ;
				if (width <= 0 || height <= 0)	continue;

				x = gItems[imageNumber][i].itemsRegions[j].items[k].iOffsetX ;
				y = gItems[imageNumber][i].itemsRegions[j].items[k].iOffsetY ;
				sx = __max(0, x);
				sy = __max(0, y);
				ex = __min(x + width, output.cols);
				ey = __min(y + height, output.rows);
				width = ex - sx;
				height = ey - sy;

				maskRoi = { sx - x,0,width,height };

				if (ex <= 0 || width <= 0)
					continue;

				maskCutted = mask(maskRoi);
				
				cv::morphologyEx(maskCutted,
					maskCutted,
					cv::MorphTypes::MORPH_ERODE,
					kernel,
					cv::Point(-1, -1),
					1, cv::BorderTypes::BORDER_CONSTANT, 0);


				for (int r = 0; r < maskCutted.rows; r++)
				{
					unsigned char *pMask = maskCutted.ptr<unsigned char>(r);

					for (int c = 0; c < maskCutted.cols; c++)
					{
						if (pMask[c] == 255)
						{
							output.at<cv::Vec3b>(y + r, x + maskRoi.tl().x + c) = color*0.4
								+ gImageSpace[imageNumber].at<cv::Vec3b>(y + r, x + maskRoi.tl().x + c)*0.6;
						}
					}
				}
			}
		}
	}

	log->info("========?éêó?ˉíê±?========");
	return 0;
}


template<typename T>
inline void setZero(T* data, int nLength)
{
	memset(data, 0, nLength * sizeof(T));
}

int Inspector::getHists(ItemHyperparam & hist, int isTop)
{
	setZero(hist.pad.hist, 768);
	setZero(hist.steel.hist, 768);
	setZero(hist.opacity.hist, 768);

	setZero(hist.pad.totalMean, 3);
	setZero(hist.pad.lowerMean, 3);
	setZero(hist.pad.upperMean, 3);
	setZero(hist.pad.lowerStdDev, 3);
	setZero(hist.pad.upperStdDev, 3);

	setZero(hist.steel.totalMean, 3);
	setZero(hist.steel.lowerMean, 3);
	setZero(hist.steel.upperMean, 3);
	setZero(hist.steel.lowerStdDev, 3);
	setZero(hist.steel.upperStdDev, 3);
	
	setZero(hist.opacity.totalMean, 3);
	setZero(hist.opacity.lowerMean, 3);
	setZero(hist.opacity.upperMean, 3);
	setZero(hist.opacity.lowerStdDev, 3);
	setZero(hist.opacity.upperStdDev, 3);

	AlgBase::emitParam(gHyperParam, isTop);
	for (int i = 0; i < gHyperParam.nPad; i++)
	{	
		for (int n = 0; n < 768; n++)
		{
			hist.pad.hist[n] += gHyperParam.pad[i].hist[n];
		}
		for (int n = 0; n < 3; n++)
		{
			hist.pad.totalMean[n] += gHyperParam.pad[i].totalMean[n];
			hist.pad.lowerMean[n] += gHyperParam.pad[i].lowerMean[n];
			hist.pad.lowerStdDev[n] += gHyperParam.pad[i].lowerStdDev[n];
			hist.pad.upperMean[n] += gHyperParam.pad[i].upperMean[n];
			hist.pad.upperStdDev[n] += gHyperParam.pad[i].upperStdDev[n];
		}
	}

	for (int n = 0; n < 768; n++)
	{
		hist.pad.hist[n] /= gHyperParam.nPad;
	}

	for (int n = 0; n < 3; n++)
	{
		hist.pad.totalMean[n] /= gHyperParam.nPad;
		hist.pad.lowerMean[n] /= gHyperParam.nPad;
		hist.pad.lowerStdDev[n] /= gHyperParam.nPad;
		hist.pad.upperMean[n] /= gHyperParam.nPad;
		hist.pad.upperStdDev[n] /= gHyperParam.nPad;
	}

	for (int i = 0; i < gHyperParam.nSteel; i++)
	{
		for (int n = 0; n < 768; n++)
		{
			hist.steel.hist[n] += gHyperParam.steel[i].hist[n];
		}
		for (int n = 0; n < 3; n++)
		{
			hist.steel.totalMean[n] += gHyperParam.steel[i].totalMean[n];
			hist.steel.lowerMean[n] += gHyperParam.steel[i].lowerMean[n];
			hist.steel.lowerStdDev[n] += gHyperParam.steel[i].lowerStdDev[n];
			hist.steel.upperMean[n] += gHyperParam.steel[i].upperMean[n];
			hist.steel.upperStdDev[n] += gHyperParam.steel[i].upperStdDev[n];
		}
	}

	for (int n = 0; n < 768; n++)
	{
		hist.steel.hist[n] /= gHyperParam.nSteel;
	}

	for (int n = 0; n < 3; n++)
	{
		hist.steel.totalMean[n] /= gHyperParam.nSteel;
		hist.steel.lowerMean[n] /= gHyperParam.nSteel;
		hist.steel.lowerStdDev[n] /= gHyperParam.nSteel;
		hist.steel.upperMean[n] /= gHyperParam.nSteel;
		hist.steel.upperStdDev[n] /= gHyperParam.nSteel;
	}

	for (int i = 0; i < gHyperParam.nOpacity; i++)
	{
		for (int n = 0; n < 768; n++)
		{
			hist.opacity.hist[n] += gHyperParam.opacity[i].hist[n];
		}
		for (int n = 0; n < 3; n++)
		{
			hist.opacity.totalMean[n] += gHyperParam.opacity[i].totalMean[n];
			hist.opacity.lowerMean[n] += gHyperParam.opacity[i].lowerMean[n];
			hist.opacity.lowerStdDev[n] += gHyperParam.opacity[i].lowerStdDev[n];
			hist.opacity.upperMean[n] += gHyperParam.opacity[i].upperMean[n];
			hist.opacity.upperStdDev[n] += gHyperParam.opacity[i].upperStdDev[n];
		}
	}

	for (int n = 0; n < 768; n++)
	{
		hist.opacity.hist[n] /= gHyperParam.nOpacity;
	}

	for (int n = 0; n < 3; n++)
	{
		hist.opacity.totalMean[n] /= gHyperParam.nOpacity;
		hist.opacity.lowerMean[n] /= gHyperParam.nOpacity;
		hist.opacity.lowerStdDev[n] /= gHyperParam.nOpacity;
		hist.opacity.upperMean[n] /= gHyperParam.nOpacity;
		hist.opacity.upperStdDev[n] /= gHyperParam.nOpacity;
	}
	return 0;
}
