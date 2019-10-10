#pragma once
#include <memory>
#include "assert.h"

template <typename T>
struct BlockNode
{
	BlockNode<T>* prev;
	BlockNode<T>* next;
	T *start;				//ָ���׵�ַ
	T *end;					//ָ��ĩβ��ַ
	int count;				//��ǰBlock����
	bool isLock = false;	//��ǰ��ַ�Ƿ���
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
		//data���ö�
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
