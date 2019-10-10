#pragma once
#include <memory>
#include "assert.h"

template <typename T>
struct BlockNode
{
	BlockNode<T>* prev;
	BlockNode<T>* next;
	T *start;				//指向首地址
	T *end;					//指向末尾地址
	int count;				//当前Block容量
	bool isLock = false;	//当前地址是否被用
};

template <typename T>
struct ListNode
{
	ListNode<T>* prev;
	ListNode<T>* next;
	T data;
	ListNode()
		:prev(nullptr),
		next(nullptr) {}
};

template <typename T>
class ListPool
{
protected:
	typedef ListNode<T> ListNode;
public:
	typedef ListNode* LinkType;
protected:
	LinkType node;
private:
	LinkType Create_Node() {
		LinkType tmp = new ListNode();
		return tmp;
	}
public:
	ListPool()
	{
		Empty_initialize();
	}
	void Empty_initialize()
	{
		node = Create_Node();
		node->prev = node;
		node->next = node;
	}
	void Remove(ListNode& link)
	{
		assert(link.prev != nullptr);
		assert(link.next != nullptr);
		//data不用动
		link.prev->next = link.next;
		link.next->prev = link.prev;

		link.next = nullptr;
		link.prev = nullptr;
	}
	void AddNode(const T& x)
	{
		LinkType tmp = Create_Node();
		tmp->data = x;
		tmp->next = node;
		tmp->prev = node->prev;
		node->prev->next = tmp;
		node->prev = tmp;
	}
	unsigned ListSize()const
	{
		unsigned t = 0;
		LinkType ptr = node->next;
		while (ptr != node)
		{
			ptr = ptr->next;
			++t;
		}
		return t;
	}
};
