#include <opencv2/opencv.hpp>
#include "Api_FPCBInspector.hpp"
#include "../intra/fpcbInspector.hpp"

#ifdef _TIME_LOG_

#endif // _TIME_LOG_


#define FPCBINSPECTOR_VERSION_MAJOR     1   // tag, big mile stone
#define FPCBINSPECTOR_VERSION_MINOR     4   // add new features or change needs
#define FPCBINSPECTOR_VERSION_REVISION  0   // fix a release bug
#define FPCBINSPECTOR_VERSION_MAINTAIN  0   // debug
#define FPCBINSPECTOR_VERSION_DATA      190927

static int imageNum = 0;
static int learnNum = 0;
static int procdNum = 0;
static int accNum = 0;

APIFPCBInspector::APIFPCBInspector()
{

}

APIFPCBInspector::~APIFPCBInspector()
{
	//api_destory();
}

static std::mutex amu;
int APIFPCBInspector::api_init(const CoarseMatchData &matchData, int isTop)
{
	/*std::cout << "matchData.gerbDPI"<<matchData.gerbDPI << std::endl;
	std::cout << "szMachineName"<<matchData.szMachineName << std::endl;
	std::cout << "maxAngleDeviation" << matchData.maxAngleDeviation << std::endl;
	std::cout << "scaleFactor" << matchData.scaleFactor << std::endl;
	std::cout << "iImgWidth" << matchData.iImgWidth << std::endl;
	std::cout << "iImgHeight" << matchData.iImgHeight << std::endl;
	std::cout << "iImgNumber" << matchData.iImgNumber << std::endl;
*/


	iniLogAlg();
	auto log = spdlog::get("log");

	std::string strFunction(__FUNCTION__);

	log->info("========" + strFunction + "========");


	int err = 0;
	g_processor = new Inspector();
	Inspector* process = (Inspector*)g_processor;
	CoarseMatchData_Input matchDataInput;
	matchDataInput >> matchData;
	imageNum = matchDataInput.iImgNumber;
	//ImageInfo output;
	//output.ptr = (uchar*)malloc(10000 * 10000 * 3);
	//cv::Mat outputMat(10000, 10000, CV_8UC3, output, 10000 * 3);
	err = process->init(matchDataInput,isTop);
	if (err != 0)
	{
		log->error("Function_Alg:" + strFunction + " Return: -1 ");
		return -1;
	}
	//api_gerber_location(output, 3);
	//process->gerberLocation(outputMat, 3);
	log->info("Function_Alg:" + strFunction + "		 Return: 0 ");
	return 0;
}

int APIFPCBInspector::api_init(string matchJson)
{
	g_processor = new Inspector();
	Inspector* process = (Inspector*)g_processor;
	//process->init(matchJson);
	return 0;
}

void APIFPCBInspector::api_getVersionNum(string &versionNum)
{
    versionNum =    std::to_string(FPCBINSPECTOR_VERSION_MAJOR) + 
                    "." + 
                    std::to_string(FPCBINSPECTOR_VERSION_MINOR) + 
                    "." + 
                    std::to_string(FPCBINSPECTOR_VERSION_REVISION) +
                    "." + 
                    std::to_string(FPCBINSPECTOR_VERSION_DATA);
}

int APIFPCBInspector::api_init_ImageNum()
{
	learnNum = 0;
	procdNum = 0;
	return 0;
}

int APIFPCBInspector::api_loadParam(string dir)
{
	if (g_processor != NULL)
	{
		Inspector *process = (Inspector*)g_processor;
		process->loadParam(dir);
	}
	return 0;
}

int APIFPCBInspector::api_destory()
{
	imageNum = 0;
	learnNum = 0;
	procdNum = 0;
	if (g_processor !=NULL )
	{
		Inspector *process = (Inspector*)g_processor;
		process->destory();
		delete g_processor;
		g_processor = NULL;
	}
	return 0;
}

int APIFPCBInspector::api_learn(ImageInfo &learnImage, int isTop)
{

	iniLogAlg();
	auto log = spdlog::get("log");

	std::string strFunction(__FUNCTION__);

	log->info("========" + strFunction + "========");


	int err = 0;
	if (learnImage.ptr == nullptr		||
		learnImage.width <= 0			||
		learnImage.height <= 0			||
		learnImage.nLabel != learnNum	||
		learnImage.step < learnImage.width)
	{
		log->error("Function_Alg:" + strFunction + " Return: -1 ");
		return -1;
	}


	learnNum = (learnNum + 1) % imageNum;
	if (g_processor != NULL)
	{
		Inspector *process = (Inspector*)g_processor;
		cv::Mat learnMat(learnImage.height, learnImage.width, CV_8UC3, learnImage.ptr, learnImage.step);
		err = process->learn(learnMat, learnImage.nLabel, isTop);
		if (err != 0)
		{
			log->error("Function_Alg:" + strFunction + " Return: -1 ");
			return -1;
		}
			
	}
	log->info("Function_Alg:" + strFunction + " Return: 0 ");
	return 0;
}

int APIFPCBInspector::api_set_configParam(const ConfigParam &configParam, int isTop)
{

    if (g_processor != NULL)
    {
        Inspector *process = (Inspector*)g_processor;

        process->setConfigparam(configParam, isTop);
    }

    return 0;
}

int APIFPCBInspector::api_get_configParam(ConfigParam &configParam, int isTop)
{

	if (g_processor != 0)
	{
		Inspector *process = (Inspector*)g_processor;

		process->getConfigparam(configParam, isTop);
	}

	return 0;
}

/*
int APIFPCBInspector::api_save_param(ConfigParam * param)
{
	if (g_processor != NULL)
	{
		Inspector *process = (Inspector*)g_processor;
		process->saveParam(param);
	}
	return 0;
}
*/

int APIFPCBInspector::api_load_space(OutputInfo *outputSpace)
{
	iniLogAlg();
	auto log = spdlog::get("log");

	std::string strFunction(__FUNCTION__);

	log->info("========" + strFunction + "========");

	int err = 0;
	if (g_processor != NULL)
	{
		Inspector *process = (Inspector*)g_processor;
		err = process->loadSpace(outputSpace);

		if (err!=0)
		{
			log->error("Function_Alg:" + strFunction + " Return: -1 ");
			return -1;
		}
		
	}
	log->info("Function_Alg: " + strFunction + " Return: 0 ");
	return err;
}


int APIFPCBInspector::api_process(ImageInfo &input, int isTop)
{
	iniLogAlg();
	auto log = spdlog::get("log");

	std::string strFunction(__FUNCTION__);

	log->info("========" + strFunction + "========");
#ifdef _TIME_LOG_
	SYSTEMTIME timeLocal;
	GetLocalTime(&timeLocal);
	
	char data[37] = { 0 };
	sprintf(data, "%04d-%02d-%02d  %02d:%02d:%02d:%02d",
		timeLocal.wYear, timeLocal.wMonth, timeLocal.wDay, 
		timeLocal.wHour, timeLocal.wMinute, timeLocal.wSecond, timeLocal.wMilliseconds);
	std::string strData = std::string(data);

	std::chrono::steady_clock::time_point timeStart = std::chrono::steady_clock::now();
	static int timeCastFull = 0;
	ofstream file;
	file.open(log_time_name, ios::out | ios::app);
	
	if (input.nLabel == 0)
	{
		if (isTop)
		{
			file << std::left << "========正面检测========" << std::endl;
		}
		else
		{
			file << std::left << "========背面检测========" << std::endl;
		}
		
	}
	file << std::left << std::setw(32) << strFunction
		<< std::right << std::setw(16) << ""
		<< std::setw(16) << ""
		<< std::setw(36) << strData
		<< std::endl;
	file.close();
#endif
	
	if (input.ptr == nullptr ||
		input.width <= 0 ||
		input.height <= 0 ||
		input.nLabel != procdNum ||
		input.step < input.width)
	{
		log->error("Function_Alg:"+ strFunction+" Return: -1 ");
		return -1;
	}
		

	procdNum = (procdNum + 1) % imageNum;
	if (g_processor != NULL)
	{
		Inspector *process = (Inspector*)g_processor;
		cv::Mat inputMat(input.height,input.width, CV_8UC3, input.ptr, input.step);
		int err= process->process(inputMat, input.nLabel, isTop);
		if (err != 0)
		{
			log->error("Function_Alg:" + strFunction + " Return: -2 ");
			return -2;
		}

	}
	if (input.nLabel >= imageNum)
	{
		accNum = 0;
	}

	log->info("Function_Alg:" + strFunction + " Return: 0 ");

	

#ifdef _TIME_LOG_
	std::chrono::steady_clock::time_point timeEnd = std::chrono::steady_clock::now();
	std::chrono::milliseconds timeCast = std::chrono::duration_cast<std::chrono::milliseconds>(timeEnd - timeStart);
	file.open(log_time_name, ios::out | ios::app);
	timeCastFull += timeCast.count();

	GetLocalTime(&timeLocal);
	sprintf(data, "%04d-%02d-%02d  %02d:%02d:%02d:%02d",
		timeLocal.wYear, timeLocal.wMonth, timeLocal.wDay,
		timeLocal.wHour, timeLocal.wMinute, timeLocal.wSecond, timeLocal.wMilliseconds);
	strData = std::string(data);
	file << std::left << std::setw(32) << strFunction 
		<< std::right << std::setw(16) << timeCastFull
		<< std::setw(16) << timeCast.count()
		<< std::setw(36) << strData
		 << std::endl;
	if (input.nLabel == imageNum -1 )
	{
		file << std::endl;
		file << std::left << "==================" << std::endl;
		file << std::endl;
		timeCastFull = 0;
	}
	else
	{
		file << std::endl;
	}
	file.close();
#endif
	return 0;
}

int APIFPCBInspector::api_process_defects(ImageInfo &input, int isTop, ConfigParam & configParam, std::vector<DefectRoi>& defects)
{
	iniLogAlg();
	auto log = spdlog::get("log");

	std::string strFunction(__FUNCTION__);

	log->info("========" + strFunction + "========");
	int err = 0;

	if (input.ptr == nullptr ||
		input.width <= 0 ||
		input.height <= 0 ||
		input.step < input.width)
	{
		log->error("Function_Alg:" + strFunction + " Return: -1 ");
		return -1;
	}
		

	if (g_processor != NULL)
	{
		Inspector *process = (Inspector*)g_processor;
		cv::Mat inputMat(input.height, input.width, CV_8UC3, input.ptr, input.step);
		err = process->process(inputMat, input.nLabel, isTop, configParam, defects);
		if (err!=0)
		{
			
			log->error("Function_Alg:" + strFunction + " Return: -1 ");
			return -1;
		}
		
	}

	log->info("Function_Alg:" + strFunction + " Return: 0 ");
	return 0;
}

int APIFPCBInspector::api_process_realtime(ConfigParam & configParam, int isTop, std::vector<DefectRoi>& defects)
{	
	iniLogAlg();
	auto log = spdlog::get("log");

	std::string strFunction(__FUNCTION__);

	log->info("========" + strFunction + "========");
	int err = 0;
	if (g_processor != NULL)
	{
		Inspector *process = (Inspector*)g_processor;
		err = process->processRealtime(configParam, isTop, defects);

		if (err != 0)
		{
			log->error("Function_Alg:" + strFunction + " Return: -1 ");
			return -1;
		}
	}
	log->info("Function_Alg:" + strFunction + " Return: 0 ");
	return 0;
}

int APIFPCBInspector::api_location(ImageInfo *output, float scale, int brushWidth)
{
	iniLogAlg();
	auto log = spdlog::get("log");

	std::string strFunction(__FUNCTION__);

	log->info("========" + strFunction + "========");

	if (output->ptr == nullptr || output->width > output->step || output->nLabel < 0)
	{
		log->error("Function_Alg:" + strFunction + " Return: -1 ");
		return -1;
	}

	int err = 0;

	scale = MIN(MAX(0.05, scale), 1.f);
	output->width = BYTEALIGNING(output->width);
	output->step = output->width * 3;
	if (g_processor != NULL)
	{
		Inspector *process = (Inspector*)g_processor;
		cv::Mat inputMat(output->height, output->width, CV_8UC3, output->ptr, output->step);
		err = process->location(inputMat, output->nLabel, scale, brushWidth);
		output->width = inputMat.cols;
		output->height = inputMat.rows;
		output->step = inputMat.step;

		if (err != 0 )
		{
			log->error("Function_Alg:" + strFunction + " Return: -1 ");
			return -1;
		}
	}
	log->info("Function_Alg:" + strFunction + " Return: 0 ");
	return 0;
}

int APIFPCBInspector::api_gerber_location(const CoarseMatchData &matchData, ImageInfo &output, int isTop, int lineWidth)
{
	iniLogAlg();
	auto log = spdlog::get("log");

	std::string strFunction(__FUNCTION__);

	log->info("========" + strFunction + "========");

	if (output.ptr == nullptr || output.width > output.step)
	{
		log->error("Function_Alg:" + strFunction + " Return: -2 ");
		return -2;
	}

    int err = 0;
    CoarseMatchData_Input matchDataInput;
    matchDataInput >> matchData;
    imageNum = matchDataInput.iImgNumber;

	//4字节对齐
	output.width = /*BYTEALIGNING*/(output.width);
	output.step = output.width * 3;
	cv::Mat outputMat(output.height, output.width, CV_8UC3, output.ptr, output.step);
	err = getArrayImg(matchDataInput,isTop, outputMat, lineWidth);

	if (err != 0)
	{
		log->error("Function_Alg:" + strFunction + "(getArrayImg) Return: "+std::to_string(err));
	}

	log->info("Function_Alg:" + strFunction + " Return: 0 ");
	return 0;
}

int APIFPCBInspector::api_processInterface(const ImageInfo & input, int imageIndex, int isTop, std::vector<DefectRoi>& defects)
{
    if (input.ptr == nullptr ||
        input.width <= 0 ||
        input.height <= 0 ||
        input.nLabel != procdNum ||
        input.step < input.width)
    {
        return -1;
    }

    if (g_processor != NULL)
    {
        Inspector *process = (Inspector*)g_processor;

        cv::Mat inputMat(input.height, input.width, CV_8UC3, input.ptr, input.step);

        int err = process->processInterface(inputMat, imageIndex, isTop, defects);

        if (err != 0)
        {
            return -2;
        }

    }
    return 0;
}

int APIFPCBInspector::api_unifyConfigparam(int isTop, Layer layer)
{
    if (g_processor != NULL)
    {
        Inspector *process = (Inspector*)g_processor;

        int err = process->unifyConfigparam(isTop, layer);

        if (err != 0)
        {
            return -2;
        }
    }
    
    return 0;
}

int APIFPCBInspector::api_getHistogram(const ImageInfo & input, int x, int y, uchar lower[3], uchar upper[3], float ch1[256], float ch2[256], float ch3[256])
{
    if (input.ptr == nullptr ||
        input.width <= 0 ||
        input.height <= 0 ||
        input.nLabel != procdNum ||
        input.step < input.width)
    {
        return -1;
    }

    if (g_processor != NULL)
    {
        Inspector *process = (Inspector*)g_processor;

        cv::Mat inputMat(input.height, input.width, CV_8UC3, input.ptr, input.step);

        int err = process->getHistogram(inputMat, x, y, lower, upper, ch1, ch2, ch3);

        if (err != 0)
        {
            return -2;
        }
    }
    return 0;
}

int APIFPCBInspector::api_setLocalParam(uchar lower[3], uchar upper[3])
{

    if (g_processor != NULL)
    {
        Inspector *process = (Inspector*)g_processor;

        int err  = process->setLocalParam(lower, upper);
        
        if (err != 0)
        {
            return -2;
        }
    }

    return 0;
}

//int APIFPCBInspector::api_fillup(const path_t *jsonList, double mouseX, double mouseY, std::vector<double>& filledX, std::vector<double>& filledY)
//{
//	fillup(jsonList, mouseX, mouseY, filledX, filledY);
//	return 0;
//}

//int APIFPCBInspector::api_doCalib(const ImageInfo & img, int iIndex, std::string szMachineName, float fAxisOffset)
//{
//	if (img.ptr == nullptr ||
//		img.width <= 0 ||
//		img.height <= 0 ||
//		img.step < img.width)
//		return -1;
//
//	cv::Mat inputMat(img.height, img.width, CV_8UC3, img.ptr, img.step);
//	doCalib(inputMat, img.nLabel, szMachineName, fAxisOffset);
//
//	return 0;
//}

//int APIFPCBInspector::api_breakup(const path_t * jsonList, const std::vector<bm::base::bmShapePoint>& mask, std::vector<std::vector<bm::base::bmShapePoint>>& ploygon)
//{
//	breakup(jsonList, mask, ploygon);
//	return 0;
//}

int APIFPCBInspector::api_getHist(ItemHyperparam & hist, int isTop)
{
	iniLogAlg();
	auto log = spdlog::get("log");

	std::string strFunction(__FUNCTION__);


	int err = 0;
	if (g_processor != NULL)
	{
		Inspector *process = (Inspector*)g_processor;
		err = process->getHists(hist, isTop);

		if (err != 0)
		{
			log->error("Function_Alg:" + strFunction + " " + std::to_string(err));
		}
	}

	log->info("Function_Alg:" + strFunction + " Return: 0 ");
	return err;
}

int APIFPCBInspector::api_observeFirstMark(const CoarseMatchData & coarseData_input, int isTop, const ImageInfo & firstSubImg_input, ImageInfo & firstSubImg_output, int extend)
{
    return 0;
}

int APIFPCBInspector::api_setTestImage(ImageInfo src, int iImgIndex, int isTop)
{
    return 0;
}

int APIFPCBInspector::api_getTestGerbCoarse(std::vector<PolyLay>&, int isTop)
{
    return 0;
}

int APIFPCBInspector::api_setTestPosition(std::vector<bm::base::bmShapePoint> gerbLTpts, std::vector<bm::base::bmShapePoint> gerbRBpts, std::vector<bm::base::bmShapePoint> imgLTpts, std::vector<bm::base::bmShapePoint> imgRBpts, int isTop)
{
    return 0;
}

int APIFPCBInspector::api_getTestGerbModify(std::vector<PolyLay>&, int isTop)
{
    return 0;
}

int APIFPCBInspector::api_doTest(int isTop)
{
    return 0;
}

//int APIFPCBInspector::api_extraGoldPad(const path_t * line, const path_t * baojiao, std::vector<std::vector<bm::base::bmShapePoint>>& vecContour)
//{
//	extraGoldPad(line, baojiao, vecContour);
//	return 0;
//}

//int APIFPCBInspector::api_getCalibData(std::string szMachineName, CalibData & data)
//{
//    getCalibData(szMachineName, data);
//    return 0;
//}
