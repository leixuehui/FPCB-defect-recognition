#pragma once

#include <vector>
#include <queue>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>
#include <stdexcept>

//class ThreadPool {
//public:
//	ThreadPool(size_t);
//	template<class F, class... Args>
//	auto enqueue(F&& f, Args&&... args)
//		->std::future<typename std::result_of<F(Args...)>::type>;
//	~ThreadPool();
//private:
//	// need to keep track of threads so we can join them
//	std::vector< std::thread > workers;
//	// the task queue
//	std::queue< std::function<void()> > tasks;
//
//	// synchronization
//	std::mutex queue_mutex;
//	std::condition_variable condition;
//	bool stop;
//};
//
//// the constructor just launches some amount of workers
//inline ThreadPool::ThreadPool(size_t threads)
//	: stop(false)
//{
//	for (size_t i = 0; i<threads; ++i)
//		workers.emplace_back(
//			[this]
//	{
//		for (;;)
//		{
//			std::function<void()> task;
//
//			{
//				std::unique_lock<std::mutex> lock(this->queue_mutex);
//				this->condition.wait(lock,
//					[this] { return this->stop || !this->tasks.empty(); });
//				if (this->stop && this->tasks.empty())
//					return;
//				task = std::move(this->tasks.front());
//				this->tasks.pop();
//			}
//
//			task();
//		}
//	}
//	);
//}
//
//// add new work item to the pool
//template<class F, class... Args>
//auto ThreadPool::enqueue(F&& f, Args&&... args)
//-> std::future<typename std::result_of<F(Args...)>::type>
//{
//	using return_type = typename std::result_of<F(Args...)>::type;
//
//	auto task = std::make_shared< std::packaged_task<return_type()> >(
//		std::bind(std::forward<F>(f), std::forward<Args>(args)...)
//		);
//
//	std::future<return_type> res = task->get_future();
//	{
//		std::unique_lock<std::mutex> lock(queue_mutex);
//
//		// don't allow enqueueing after stopping the pool
//		if (stop)
//			throw std::runtime_error("enqueue on stopped ThreadPool");
//
//		tasks.emplace([task]() { (*task)(); });
//	}
//	condition.notify_one();
//	return res;
//}
//
//// the destructor joins all threads
//inline ThreadPool::~ThreadPool()
//{
//	{
//		std::unique_lock<std::mutex> lock(queue_mutex);
//		stop = true;
//	}
//	condition.notify_all();
//	for (std::thread &worker : workers)
//		worker.join();
//}

///////////////////////////////////////////////////////////////////////////
//thread pool
///////////////////////////////////////////////////////////////////////////

class JoinThreads
{
	std::vector<std::thread>& threads;
public:
	explicit JoinThreads(std::vector<std::thread>& threads_) :threads(threads_) {}
	~JoinThreads()
	{
		for (int t = 0; t < threads.size(); t++)
		{
			if (threads[t].joinable())	threads[t].join();
		}
	}
};

template<typename T>
class ThreadSafeQueue
{
private:
	mutable std::mutex mu;
	std::queue<T> dataQueue;
	std::condition_variable dataCond;
public:
	ThreadSafeQueue() {}
	void push(T newData)
	{
		std::lock_guard<std::mutex> lk(mu);
		dataQueue.push(std::move(newData));
		dataCond.notify_one();
	}
	void waitPop(T &value)
	{
		std::unique_lock<std::mutex> lk(mu);
		dataCond.wait(lk, [this] {return !dataQueue.empty(); });
		value = std::move(dataQueue.front());
		dataQueue.pop();
	}
	std::shared_ptr<T> waitPop()
	{
		std::unique_lock<std::mutex> lk(mu);
		dataCond.wait(lk, [this] {return !dataQueue.empty(); });
		std::shared_ptr<T> res(make_shared<T>(std::move(dataQueue.front())));
		dataQueue.pop();
		return res;
	}
	bool tryPop(T &value)
	{
		std::lock_guard<std::mutex> lk(mu);
		if (dataQueue.empty())	return false;
		value = std::move(dataQueue.front());
		dataQueue.pop();
		return true;
	}
	std::shared_ptr<T> tryPop()
	{
		std::lock_guard < std::mutex > lk(mu);
		if (dataQueue.empty())	return false;
		std::shared_ptr<T> res(make_shared<T>(std::move(dataQueue.front())));
		dataQueue.pop();
		return true;
	}
	bool empty() const
	{
		std::lock_guard<std::mutex> lk(mu);
		return dataQueue.empty();
	}
};

class ThreadPool
{
	using Task = std::function<void()>;
	std::atomic_bool done;
	ThreadSafeQueue<Task> workQueue;
	std::vector<std::thread> threads;
	JoinThreads joiner;
	void workerThread()
	{
		while (!done)//while(!done)ª•≥‚ ß∞‹
		{
			Task task;
			//if (done && workQueue.empty()) return;
			if (workQueue.tryPop(task))	task();
			else						std::this_thread::yield();
			
			Sleep(1);
		}
	}
public:
	ThreadPool() :done(false), joiner(threads)
	{
		unsigned const threadCount = std::thread::hardware_concurrency() - 1;
		try
		{
			for (unsigned i = 0; i < threadCount; i++)
			{
				threads.emplace_back(std::thread(&ThreadPool::workerThread, this));
			}
		}
		catch (...)
		{
			done = true;
			throw;
		}
	}

	~ThreadPool()
	{
		done = true;
	}

	template<class Func, class... T>
	auto submit(Func&& func, T&&... param)
		-> std::future<typename std::result_of<Func(T...)>::type>
	{
		using ret = typename std::result_of<Func(T...)>::type;
		auto task = std::make_shared<std::packaged_task<ret()>>(std::bind(std::forward<Func>(func), std::forward<T>(param)...));
		std::future<ret> res = task->get_future();
		workQueue.push([task]() {(*task)(); });
		return res;
	}
};