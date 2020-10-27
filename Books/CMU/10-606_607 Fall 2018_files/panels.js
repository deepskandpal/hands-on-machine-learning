//Navbar Box Shadow on Scroll 
$(function(){
    var navbar = $('.navbar');
    $(window).scroll(function(){
        if($(window).scrollTop() <= 40){
       	    navbar.css('box-shadow', 'none');
        } else {
            navbar.css('box-shadow', '0px 10px 20px rgba(0, 0, 0, 0.4)'); 
        }
    });  
})


//Function from Bluthemes, lets you add li elemants to affix object without having to alter and data attributes set out by bootstrap
$(function(){

    // name your elements here
    var stickyElement   = '.panel-affix'; // the element you want to make sticky
    var bottomElement   = '#footer'; // the bottom element where you want the sticky element to stop (usually the footer) 

    // make sure the element exists on the page before trying to initalize
    if($( stickyElement ).length){
	$( stickyElement ).each(function(){

	    // let's save some messy code in clean variables
	    // when should we start affixing? (the amount of pixels to the top from the element)
	    var fromTop = $( this ).offset().top;
	    // where is the bottom of the element?
	    var fromBottom = $( document ).height()-($( this ).offset().top + $( this ).outerHeight());
	    // where should we stop? (the amount of pixels from the top where the bottom element is)
	    // also add the outer height mismatch to the height of the element to account for padding and borders
	    var stopOn = $( document ).height()-( $( bottomElement ).offset().top)+($( this ).outerHeight() - $( this ).height()); 

	    // if the element doesn't need to get sticky, then skip it so it won't mess up your layout
	    if( (fromBottom-stopOn) > 200 ){
		// let's put a sticky width on the element and assign it to the top
		$( this ).css('width', $( this ).width()).css('top', 0).css('position', '');
		// assign the affix to the element
		$( this ).affix({
		    offset: { 
		        // make it stick where the top pixel of the element is
		        top: fromTop - 80,  
		        // make it stop where the top pixel of the bottom element is
		        bottom: stopOn
		    }
		    // when the affix get's called then make sure the position is the default (fixed) and it's at the top
		}).on('affix.bs.affix', function(){ $( this ).css('top', '80px').css('position', ''); });
	    }
	    // trigger the scroll event so it always activates 
	    $( window ).trigger('scroll'); 
	}); 
    }

});

//Smooth Scrolling For Internal Page Links
$(function() {
    $('a[href*=#]:not([href=#])').not('[data-toggle="collapse"]').click(function() {
	if (location.pathname.replace(/^\//,'') == this.pathname.replace(/^\//,'') && location.hostname == this.hostname) {
	    var target = $(this.hash);
	    target = target.length ? target : $('[name=' + this.hash.slice(1) +']');
	    if (target.length) {
		$('html,body').animate({
		    scrollTop: (target.offset().top)
		}, 1000);
		return false;
	    }
	}
    });
});
