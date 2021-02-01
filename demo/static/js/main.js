// variables
let userName = null;
let state = 'SUCCESS';

// functions
function Message(arg)
{
    this.text = arg.text;
    this.message_side = arg.message_side;

    this.draw = function (_this)
    {
        return function ()
        {
            let $message;
            $message = $($('.message_template').clone().html());
            $message.addClass(_this.message_side).find('.text').html(_this.text);
            $('.messages').append($message);

            return setTimeout(function () {
                return $message.addClass('appeared');
            }, 0);
        };
    }(this);
    return this;
}

function getMessageText()
{
    let $message_input;
    $message_input = $('.message_input');
    return $message_input.val();
}

function sendMessage(text, message_side)
{
    let $messages, message;
    $('.message_input').val('');
    $messages = $('.messages');
    message = new Message({
        text: text,
        message_side: message_side
    });
    message.draw();
    $messages.animate({scrollTop: $messages.prop('scrollHeight')}, 300);
}

function greet()
{
    setTimeout(function ()
    {
        return sendMessage("스마트홈 데모에 오신걸 환영합니다.", 'left');
    }, 100);

    setTimeout(function ()
    {
        return sendMessage("성함을 알려주세요.", 'left');
    }, 101);
}

function onClickAsEnter(e)
{
    if (e.keyCode === 13)
    {
        onSendButtonClicked()
    }
}

function setUserName(username)
{
    if (username != null && username.replace(" ", "" !== ""))
    {
        setTimeout(function ()
        {
            return sendMessage("반갑습니다." + username + "님. 닉네임이 설정되었습니다.", 'left');
        }, 100);
        setTimeout(function ()
        {
            return sendMessage("스마트홈 데모는 사용자의 질문 의도를 분류합니다. 분류되는 의도는 다음과 같습니다. '난방, 주차 위치, 가스 밸브, 조명, 방범, 환기, 날씨, 간단한 인사, 검색'", 'left');
        }, 101);
        setTimeout(function ()
        {
            return sendMessage("무엇이든 물어보세요!", 'left');
        }, 102);
        return username;
    }
    else
    {
        setTimeout(function ()
        {
            return sendMessage("올바른 닉네임을 이용해주세요.", 'left');
        }, 1000);

        return null;
    }
}

//http://127.0.0.1:8080/request_chat/%EC%9D%B4%EC%A0%95%EB%AF%BC/%ED%95%98%ED%95%98
function requestChat(messageText, url_pattern)
{
//    console.log("requestChat")
//    console.log(url_pattern)
//    console.log(messageText)
//    console.log("http://127.0.0.1:8080/" + url_pattern + '/' + userName + '/' + messageText,)
    $.ajax({
        //url: "http://127.0.0.1:8080/" + url_pattern + '/' + userName + '/' + messageText,
        url: "/" + url_pattern + '/' + userName + '/' + messageText,

        type: "GET",
        dataType: "json",
        success: function (data)
        {
//            console.log('여기')
//            // 인텐트 출력  1월 21일
            setTimeout(function ()
            {
                intent = data['intent'];
                answer_tmp = sendMessage(intent, 'left');

                return answer_tmp
            }, 100);

            setTimeout(function ()
            {
                score = data['score'];
                answer_tmp = sendMessage(score, 'left');

                return answer_tmp
            }, 101);

//            intent = data['intent'];
//            answer_tmp = sendMessage(intent, 'left');
//            return answer_tmp
            //

//            state = data['state'];
//
//            if (state === 'SUCCESS')
//            {
//                answer_tmp = sendMessage(data['answer'], 'left');
//                return answer_tmp
//            }
//            else if (state === 'REQUIRE_LOCATION')
//            {
//                return sendMessage('어느 지역을 알려드릴까요?', 'left');
//            }
//            else
//            {
//                return sendMessage('죄송합니다. 무슨말인지 잘 모르겠어요.', 'left');
//            }
        },

        error: function (request, status, error)
        {
            console.log(error);

            return sendMessage('죄송합니다. 서버 연결에 실패했습니다.', 'left');
        }
    });
}

function onSendButtonClicked()
{
    let messageText = getMessageText();
    sendMessage(messageText, 'right');

    if (userName == null)
    {
        userName = setUserName(messageText);
    }
    else
    {
        if (state.includes('REQUIRE'))
        {
            return requestChat(messageText, 'fill_slot');
        }
        else
        {
            return requestChat(messageText, 'request_chat');
        }
//        if (messageText.includes('안녕'))
//        {
//            setTimeout(function ()
//            {
//                return sendMessage("안녕하세요. 저는 Kochat 여행봇입니다.", 'left');
//            }, 101);
//        }
//        else if (messageText.includes('고마워'))
//        {
//            setTimeout(function ()
//            {
//                return sendMessage("천만에요. 더 물어보실 건 없나요?", 'left');
//            }, 102);
//        }
//        else if (messageText.includes('없어'))
//        {
//            setTimeout(function ()
//            {
//                return sendMessage("그렇군요. 알겠습니다!", 'left');
//            }, 103);
//
//        }
//        else if (state.includes('REQUIRE'))
//        {
//            return requestChat(messageText, 'fill_slot');
//        }
//        else
//        {
//            return requestChat(messageText, 'request_chat');
//        }
    }
}


///////////////////////////////////////////////////////////////////
// html에서 버튼을 누르면, python 'eorrect_func' 함수 실행
function correctButtonClicked()
{
//    let messageText = getMessageText();
    return requestConrrect('request_correct');
}

function requestConrrect(url_pattern)
{
//    console.log("---")
//    console.log("http://127.0.0.1:8080/" + url_pattern + '/' + userName + '/' + messageText,)
    $.ajax({
//        url: "http://127.0.0.1:8080/" + url_pattern + '/' + userName,
        url: "/" + url_pattern + '/' + userName,
        type: "GET",
        dataType: "json",
        success: function (data)
        {
            text1 = data['input']
            console.log(text1);
            return sendMessage(text1, 'right');
//            return data

        },

        error: function (request, status, error)
        {
            console.log(error);

            return sendMessage('죄송합니다. 서버 연결에 실패했습니다.', 'left');
        }
    });
}

function inCorrectButtonClicked()
{
    return requestInconrrect('request_incorrect');
}

function requestInconrrect(url_pattern)
{
//    console.log("---")
//    console.log("http://127.0.0.1:8080/" + url_pattern + '/' + userName + '/' + messageText,)
    $.ajax({
//        url: "http://127.0.0.1:8080/" + url_pattern + '/' + userName,
        url: "/" + url_pattern + '/' + userName,
        type: "GET",
        dataType: "json",
        success: function (data)
        {
            text1 = data['input']
            console.log(text1);
            return sendMessage(text1, 'right');
//            return data

        },

        error: function (request, status, error)
        {
            console.log(error);

            return sendMessage('죄송합니다. 서버 연결에 실패했습니다.', 'left');
        }
    });
}